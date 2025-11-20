# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from importlib.metadata import version

from pydantic import BaseModel

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.external import load_external_apis
from llama_stack.core.server.router_registry import build_router, has_router
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
    run_config: StackRunConfig


async def get_provider_impl(config, deps):
    impl = DistributionInspectImpl(config, deps)
    await impl.initialize()
    return impl


class DistributionInspectImpl(Inspect):
    def __init__(self, config: DistributionInspectConfig, deps):
        self.config = config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def list_routes(self, api_filter: str | None = None) -> ListRoutesResponse:
        run_config: StackRunConfig = self.config.run_config

        # Helper function to determine if a route should be included based on api_filter
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

        ret = []
        external_apis = load_external_apis(run_config)
        all_endpoints = get_all_api_routes(external_apis)

        # Helper function to get provider types for an API
        def get_provider_types(api: Api) -> list[str]:
            if api.value in ["providers", "inspect"]:
                return []  # These APIs don't have "real" providers  they're internal to the stack
            providers = run_config.providers.get(api.value, [])
            return [p.provider_type for p in providers] if providers else []

        # Process webmethod-based routes (legacy)
        for api, endpoints in all_endpoints.items():
            # Skip APIs that have routers - they'll be processed separately
            if has_router(api):
                continue

            provider_types = get_provider_types(api)
            # Always include provider and inspect APIs, filter others based on run config
            if api.value in ["providers", "inspect"] or provider_types:
                ret.extend(
                    [
                        RouteInfo(
                            route=e.path,
                            method=next(iter([m for m in e.methods if m != "HEAD"])),
                            provider_types=provider_types,
                        )
                        for e, webmethod in endpoints
                        if e.methods is not None and should_include_route(webmethod)
                    ]
                )

        # Helper function to determine if a router route should be included based on api_filter
        def should_include_router_route(route, router_prefix: str | None) -> bool:
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

        # Process router-based routes
        def dummy_impl_getter(api: Api) -> None:
            """Dummy implementation getter for route inspection."""
            return None

        from llama_stack.core.resolver import api_protocol_map

        protocols = api_protocol_map(external_apis)
        for api in protocols.keys():
            if not has_router(api):
                continue

            router = build_router(api, dummy_impl_getter)
            if not router:
                continue

            provider_types = get_provider_types(api)
            # Only include if there are providers (or it's a special API)
            if api.value in ["providers", "inspect"] or provider_types:
                router_prefix = getattr(router, "prefix", None)
                for route in router.routes:
                    # Extract HTTP methods from the route
                    # FastAPI routes have methods as a set
                    if hasattr(route, "methods") and route.methods:
                        methods = {m for m in route.methods if m != "HEAD"}
                        if methods and should_include_router_route(route, router_prefix):
                            # FastAPI already combines router prefix with route path
                            path = route.path

                            ret.append(
                                RouteInfo(
                                    route=path,
                                    method=next(iter(methods)),
                                    provider_types=provider_types,
                                )
                            )

        return ListRoutesResponse(data=ret)

    async def health(self) -> HealthInfo:
        return HealthInfo(status=HealthStatus.OK)

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))

    async def shutdown(self) -> None:
        pass
