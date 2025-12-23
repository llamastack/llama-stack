# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from importlib.metadata import version
from typing import Any

from pydantic import BaseModel

from llama_stack.core.datatypes import StackConfig
from llama_stack.core.external import load_external_apis
from llama_stack.core.server.fastapi_router_registry import (
    _ROUTER_FACTORIES,
    build_fastapi_router,
    get_router_routes,
)
from llama_stack.core.server.routes import get_all_api_routes
from llama_stack.core.utils.config import redact_sensitive_fields
from llama_stack.log import get_logger
from llama_stack_api import (
    Admin,
    Api,
    HealthInfo,
    HealthResponse,
    HealthStatus,
    InspectProviderRequest,
    ListProvidersResponse,
    ListRoutesRequest,
    ListRoutesResponse,
    ProviderInfo,
    RouteInfo,
    VersionInfo,
)

logger = get_logger(name=__name__, category="core")


class AdminImplConfig(BaseModel):
    config: StackConfig


async def get_provider_impl(config, deps):
    impl = AdminImpl(config, deps)
    await impl.initialize()
    return impl


class AdminImpl(Admin):
    def __init__(self, config: AdminImplConfig, deps):
        self.config = config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        logger.debug("AdminImpl.shutdown")
        pass

    # Provider management methods
    async def list_providers(self) -> ListProvidersResponse:
        config = self.config.config
        safe_config = StackConfig(**redact_sensitive_fields(config.model_dump()))
        providers_health = await self.get_providers_health()
        ret = []
        for api, providers in safe_config.providers.items():
            for p in providers:
                # Skip providers that are not enabled
                if p.provider_id is None:
                    continue
                ret.append(
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        config=p.config,
                        health=providers_health.get(api, {}).get(
                            p.provider_id,
                            HealthResponse(
                                status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                            ),
                        ),
                    )
                )

        return ListProvidersResponse(data=ret)

    async def inspect_provider(self, request: InspectProviderRequest) -> ProviderInfo:
        all_providers = await self.list_providers()
        for p in all_providers.data:
            if p.provider_id == request.provider_id:
                return p

        raise ValueError(f"Provider {request.provider_id} not found")

    async def get_providers_health(self) -> dict[str, dict[str, HealthResponse]]:
        """Get health status for all providers.

        Returns:
            Dict[str, Dict[str, HealthResponse]]: A dictionary mapping API names to provider health statuses.
                Each API maps to a dictionary of provider IDs to their health responses.
        """
        providers_health: dict[str, dict[str, HealthResponse]] = {}

        # The timeout has to be long enough to allow all the providers to be checked, especially in
        # the case of the inference router health check since it checks all registered inference
        # providers.
        # The timeout must not be equal to the one set by health method for a given implementation,
        # otherwise we will miss some providers.
        timeout = 3.0

        async def check_provider_health(impl: Any) -> tuple[str, HealthResponse] | None:
            # Skip special implementations (inspect/providers/admin) that don't have provider specs
            if not hasattr(impl, "__provider_spec__"):
                return None
            api_name = impl.__provider_spec__.api.name
            if not hasattr(impl, "health"):
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                    ),
                )

            try:
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                return api_name, health
            except TimeoutError:
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.ERROR, message=f"Health check timed out after {timeout} seconds"
                    ),
                )
            except Exception as e:
                return (
                    api_name,
                    HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"),
                )

        # Create tasks for all providers
        tasks = [check_provider_health(impl) for impl in self.deps.values()]

        # Wait for all health checks to complete
        results = await asyncio.gather(*tasks)

        # Organize results by API and provider ID
        for result in results:
            if result is None:  # Skip special implementations
                continue
            api_name, health_response = result
            providers_health[api_name] = health_response

        return providers_health

    # Inspect methods
    async def list_routes(self, request: ListRoutesRequest) -> ListRoutesResponse:
        config: StackConfig = self.config.config
        api_filter = request.api_filter

        # Helper function to determine if a route should be included based on api_filter
        # TODO: remove this once we've migrated all APIs to FastAPI routers
        def should_include_route(webmethod) -> bool:
            if api_filter is None:
                # Default: only non-deprecated APIs
                return not webmethod.deprecated
            elif api_filter == "deprecated":
                # Special filter: show deprecated routes regardless of their actual level
                return bool(webmethod.deprecated)
            else:
                # Filter by API level (non-deprecated routes only)
                return not webmethod.deprecated and webmethod.level == api_filter

        # Helper function to get provider types for an API
        def _get_provider_types(api: Api) -> list[str]:
            if api.value in ["providers", "inspect", "admin"]:
                return []  # These APIs don't have "real" providers - they're internal to the stack
            providers = config.providers.get(api.value, [])
            return [p.provider_type for p in providers] if providers else []

        # Helper function to determine if a router route should be included based on api_filter
        def _should_include_router_route(route, router_prefix: str | None) -> bool:
            """Check if a router-based route should be included based on api_filter."""
            # Check deprecated status
            route_deprecated = getattr(route, "deprecated", False) or False

            if api_filter is None:
                # Default: only non-deprecated routes
                return not route_deprecated
            elif api_filter == "deprecated":
                # Special filter: show deprecated routes regardless of their actual level
                return route_deprecated
            else:
                # Filter by API level (non-deprecated routes only)
                # Extract level from router prefix (e.g., "/v1" -> "v1")
                if router_prefix:
                    prefix_level = router_prefix.lstrip("/")
                    return not route_deprecated and prefix_level == api_filter
                return not route_deprecated

        ret = []
        external_apis = load_external_apis(config)
        all_endpoints = get_all_api_routes(external_apis)

        # Process routes from APIs with FastAPI routers
        for api_name in _ROUTER_FACTORIES.keys():
            api = Api(api_name)
            router = build_fastapi_router(api, None)  # we don't need the impl here, just the routes
            if router:
                router_routes = get_router_routes(router)
                for route in router_routes:
                    if _should_include_router_route(route, router.prefix):
                        if route.methods is not None:
                            available_methods = [m for m in route.methods if m != "HEAD"]
                            if available_methods:
                                ret.append(
                                    RouteInfo(
                                        route=route.path,
                                        method=available_methods[0],
                                        provider_types=_get_provider_types(api),
                                    )
                                )

        # Process routes from legacy webmethod-based APIs
        for api, endpoints in all_endpoints.items():
            # Skip APIs that have routers (already processed above)
            if api.value in _ROUTER_FACTORIES:
                continue

            # Always include provider, inspect, and admin APIs, filter others based on run config
            if api.value in ["providers", "inspect", "admin"]:
                ret.extend(
                    [
                        RouteInfo(
                            route=e.path,
                            method=next(iter([m for m in e.methods if m != "HEAD"])),
                            provider_types=[],  # These APIs don't have "real" providers - they're internal to the stack
                        )
                        for e, webmethod in endpoints
                        if e.methods is not None and should_include_route(webmethod)
                    ]
                )
            else:
                providers = config.providers.get(api.value, [])
                if providers:  # Only process if there are providers for this API
                    ret.extend(
                        [
                            RouteInfo(
                                route=e.path,
                                method=next(iter([m for m in e.methods if m != "HEAD"])),
                                provider_types=[p.provider_type for p in providers],
                            )
                            for e, webmethod in endpoints
                            if e.methods is not None and should_include_route(webmethod)
                        ]
                    )

        return ListRoutesResponse(data=ret)

    async def health(self) -> HealthInfo:
        return HealthInfo(status=HealthStatus.OK)

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))
