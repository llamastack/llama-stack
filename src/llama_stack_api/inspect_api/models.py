# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Inspect API requests and responses.

This module defines the request and response models for the Inspect API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Literal

from pydantic import BaseModel, Field

from llama_stack_api.datatypes import HealthStatus
from llama_stack_api.schema_utils import json_schema_type

# Valid values for the route filter parameter.
# Actual API levels: v1, v1alpha, v1beta (filters by level, excludes deprecated)
# Special filter value: "deprecated" (shows deprecated routes regardless of level)
ApiFilter = Literal["v1", "v1alpha", "v1beta", "deprecated"]


@json_schema_type
class RouteInfo(BaseModel):
    """Information about an API route including its path, method, and implementing providers."""

    route: str = Field(description="The API route path")
    method: str = Field(description="The HTTP method for the route")
    provider_types: list[str] = Field(description="List of provider types implementing this route")


@json_schema_type
class HealthInfo(BaseModel):
    """Health status information for the service."""

    status: HealthStatus = Field(description="The health status of the service")


@json_schema_type
class VersionInfo(BaseModel):
    """Version information for the service."""

    version: str = Field(description="The version string of the service")


@json_schema_type
class ListRoutesResponse(BaseModel):
    """Response containing a list of all available API routes."""

    data: list[RouteInfo] = Field(description="List of available API routes")
