# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.version import (
    LLAMA_STACK_API_V1,
)
from llama_stack.providers.datatypes import HealthStatus
from llama_stack.schema_utils import json_schema_type, webmethod

# Valid values for the route filter parameter.
# Actual API levels: v1, v1alpha, v1beta (filters by level, excludes deprecated)
# Special filter value: "deprecated" (shows deprecated routes regardless of level)
ApiFilter = Literal["v1", "v1alpha", "v1beta", "deprecated"]


@json_schema_type
class RouteInfo(BaseModel):
    """Information about an API route including its path, method, and implementing providers.

    :param route: The API endpoint path
    :param method: HTTP method for the route
    :param provider_types: List of provider types that implement this route
    """

    route: str
    method: str
    provider_types: list[str]


@json_schema_type
class HealthInfo(BaseModel):
    """Health status information for the service.

    :param status: Current health status of the service
    """

    status: HealthStatus


@json_schema_type
class VersionInfo(BaseModel):
    """Version information for the service.

    :param version: Version number of the service
    """

    version: str


class ListRoutesResponse(BaseModel):
    """Response containing a list of all available API routes.

    :param data: List of available route information objects
    """

    data: list[RouteInfo]


@runtime_checkable
class Inspect(Protocol):
    """Inspect

    APIs for inspecting the Llama Stack service, including health status, available API routes with methods and implementing providers.
    """

    @webmethod(route="/inspect/routes", method="GET", level=LLAMA_STACK_API_V1)
    async def list_routes(self, api_filter: ApiFilter | None = None) -> ListRoutesResponse:
        """List routes.

        List all available API routes with their methods and implementing providers.

        :param api_filter: Optional filter to control which routes are returned. Can be an API level ('v1', 'v1alpha', 'v1beta') to show non-deprecated routes at that level, or 'deprecated' to show deprecated routes across all levels. If not specified, returns all non-deprecated routes.
        :returns: Response containing information about all available routes.
        """
        ...

    @webmethod(route="/health", method="GET", level=LLAMA_STACK_API_V1, require_authentication=False)
    async def health(self) -> HealthInfo:
        """Get health status.

        Get the current health status of the service.

        :returns: Health information indicating if the service is operational.
        """
        ...

    @webmethod(route="/version", method="GET", level=LLAMA_STACK_API_V1, require_authentication=False)
    async def version(self) -> VersionInfo:
        """Get version.

        Get the version of the service.

        :returns: Version information containing the service version number.
        """
        ...
