# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from importlib.metadata import version

from pydantic import BaseModel

from llama_stack.core.datatypes import StackConfig
from llama_stack.core.distribution import builtin_automatically_routed_apis
from llama_stack.core.external import load_external_apis
from llama_stack.core.server.fastapi_router_registry import (
    _ROUTER_FACTORIES,
    build_fastapi_router,
    get_router_routes,
)
from llama_stack.core.server.routes import get_all_api_routes
from llama_stack_api import (
    Api,
    HealthInfo,
    HealthStatus,
    Inspect,
    ListRoutesResponse,
    RouteInfo,
    VersionInfo,
)


class DistributionInspectConfig(BaseModel):
    config: StackConfig


async def get_provider_impl(config, deps):
    impl = DistributionInspectImpl(config, deps)
    await impl.initialize()
    return impl


class DistributionInspectImpl(Inspect):
    def __init__(self, config: DistributionInspectConfig, deps):
        self.stack_config = config.config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def list_routes(self, api_filter: str | None = None) -> ListRoutesResponse:
        config: StackConfig = self.stack_config

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
            if api.value in ["providers", "inspect"]:
                return []  # These APIs don't have "real" providers  they're internal to the stack

            # For routing table APIs, look up providers from their router API
            # (e.g., benchmarks -> eval, models -> inference, etc.)
            auto_routed_apis = builtin_automatically_routed_apis()
            for auto_routed in auto_routed_apis:
                if auto_routed.routing_table_api == api:
                    # This is a routing table API, use its router API for providers
                    providers = config.providers.get(auto_routed.router_api.value, [])
                    return [p.provider_type for p in providers] if providers else []

            # Regular API, look up providers directly
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

            # Always include provider and inspect APIs, filter others based on run config
            if api.value in ["providers", "inspect"]:
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

    async def shutdown(self) -> None:
        pass
