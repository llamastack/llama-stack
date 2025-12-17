# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Providers API requests and responses.

This module defines the request and response models for the Providers API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api.datatypes import HealthResponse
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class ProviderInfo(BaseModel):
    """Information about a registered provider including its configuration and health status."""

    api: str = Field(..., description="The API name this provider implements")
    provider_id: str = Field(..., description="Unique identifier for the provider")
    provider_type: str = Field(..., description="The type of provider implementation")
    config: dict[str, Any] = Field(..., description="Configuration parameters for the provider")
    health: HealthResponse = Field(..., description="Current health status of the provider")


@json_schema_type
class ListProvidersResponse(BaseModel):
    """Response containing a list of all available providers."""

    data: list[ProviderInfo] = Field(..., description="List of provider information objects")


@json_schema_type
class InspectProviderRequest(BaseModel):
    """Request model for inspecting a provider by ID."""

    provider_id: str = Field(..., description="The ID of the provider to inspect.")
