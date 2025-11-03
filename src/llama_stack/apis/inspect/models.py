# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.providers.datatypes import HealthStatus
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class RouteInfo(BaseModel):
    """Information about an API route including its path, method, and implementing providers."""

    route: str = Field(..., description="The API endpoint path")
    method: str = Field(..., description="HTTP method for the route")
    provider_types: list[str] = Field(..., description="List of provider types that implement this route")


@json_schema_type
class HealthInfo(BaseModel):
    """Health status information for the service."""

    status: HealthStatus = Field(..., description="Current health status of the service")


@json_schema_type
class VersionInfo(BaseModel):
    """Version information for the service."""

    version: str = Field(..., description="Version number of the service")


class ListRoutesResponse(BaseModel):
    """Response containing a list of all available API routes."""

    data: list[RouteInfo] = Field(..., description="List of available route information objects")
