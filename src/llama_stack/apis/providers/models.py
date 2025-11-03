# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.datatypes import HealthResponse
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ProviderInfo(BaseModel):
    """Information about a registered provider including its configuration and health status."""

    api: str = Field(..., description="The API name this provider implements")
    provider_id: str = Field(..., description="Unique identifier for the provider")
    provider_type: str = Field(..., description="The type of provider implementation")
    config: dict[str, Any] = Field(..., description="Configuration parameters for the provider")
    health: HealthResponse = Field(..., description="Current health status of the provider")


class ListProvidersResponse(BaseModel):
    """Response containing a list of all available providers."""

    data: list[ProviderInfo] = Field(..., description="List of provider information objects")
