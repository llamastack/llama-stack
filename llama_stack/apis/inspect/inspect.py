# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.providers.datatypes import HealthStatus
from llama_stack.schema_utils import json_schema_type, webmethod


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
    @webmethod(route="/inspect/routes", method="GET")
    async def list_routes(self) -> ListRoutesResponse:
        """List all available API routes with their methods and implementing providers.

        :returns: Response containing information about all available routes.
        """
        ...

    @webmethod(route="/health", method="GET")
    async def health(self) -> HealthInfo:
        """Get the current health status of the service.

        :returns: Health information indicating if the service is operational.
        """
        ...

    @webmethod(route="/version", method="GET")
    async def version(self) -> VersionInfo:
        """Get the version of the service.

        :returns: Version information containing the service version number.
        """
        ...
