# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from importlib.metadata import version

from pydantic import BaseModel

from llama_stack.apis.inspect import (
    HealthInfo,
    Inspect,
    ListRoutesResponse,
    RouteInfo,
    VersionInfo,
)
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.external import load_external_apis
from llama_stack.core.resolver import api_protocol_map
from llama_stack.core.server.routers import create_router, has_router
from llama_stack.providers.datatypes import Api, HealthStatus


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
                # Default: only non-deprecated v1 APIs
                return not webmethod.deprecated and webmethod.level == LLAMA_STACK_API_V1
            elif api_filter == "deprecated":
                # Special filter: show deprecated routes regardless of their actual level
                return bool(webmethod.deprecated)
            else:
                # Filter by API level (non-deprecated routes only)
                return not webmethod.deprecated and webmethod.level == api_filter

        ret = []

        # Create a dummy impl_getter for router creation
        def dummy_impl_getter(_api: Api) -> None:
            return None

        # Get all APIs that should be served

        external_apis = load_external_apis(run_config)
        protocols = api_protocol_map(external_apis)

        # Get APIs to serve
        if run_config.apis:
            apis_to_serve = set(run_config.apis)
        else:
            apis_to_serve = set(protocols.keys())

        apis_to_serve.add("inspect")
        apis_to_serve.add("providers")
        apis_to_serve.add("prompts")
        apis_to_serve.add("conversations")

        # Get routes from routers
        for api_str in apis_to_serve:
            api = Api(api_str)

            # Skip if no router registered
            if not has_router(api):
                continue

            # Create router to extract routes
            router = create_router(api, dummy_impl_getter)
            if not router:
                continue

            # Extract routes from the router
            provider_types: list[str] = []
            if api.value in ["providers", "inspect"]:
                # These APIs don't have "real" providers
                provider_types = []
            else:
                providers = run_config.providers.get(api.value, [])
                provider_types = [p.provider_type for p in providers]

            # Extract routes from router
            for route in router.routes:
                if not hasattr(route, "path") or not hasattr(route, "methods"):
                    continue

                # Filter out HEAD method
                methods = [m for m in route.methods if m != "HEAD"]
                if not methods:
                    continue

                # Get full path (prefix + path)
                path = route.path
                if hasattr(router, "prefix") and router.prefix:
                    if path.startswith("/"):
                        full_path = path
                    else:
                        full_path = router.prefix + "/" + path
                        full_path = full_path.replace("//", "/")
                else:
                    full_path = path

                ret.append(
                    RouteInfo(
                        route=full_path,
                        method=methods[0],
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
