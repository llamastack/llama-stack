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
from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.external import load_external_apis
from llama_stack.core.server.routes import get_all_api_routes
from llama_stack.providers.datatypes import HealthStatus


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
        for api, endpoints in all_endpoints.items():
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
                providers = run_config.providers.get(api.value, [])
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
