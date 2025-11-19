# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack_api.datatypes import HealthResponse
from llama_stack_api.schema_utils import json_schema_type, webmethod
from llama_stack_api.version import LLAMA_STACK_API_V1


@json_schema_type
class ProviderInfo(BaseModel):
    """Information about a registered provider including its configuration and health status.

    :param api: The API name this provider implements
    :param provider_id: Unique identifier for the provider
    :param provider_type: The type of provider implementation
    :param config: Configuration parameters for the provider
    :param health: Current health status of the provider
    """

    api: str
    provider_id: str
    provider_type: str
    config: dict[str, Any]
    health: HealthResponse


@json_schema_type
class ListProvidersResponse(BaseModel):
    """Response containing a list of all available providers.

    :param data: List of provider information objects
    """

    data: list[ProviderInfo]


@runtime_checkable
class Providers(Protocol):
    """Providers

    Providers API for inspecting, listing, and modifying providers and their configurations.
    """

    @webmethod(route="/providers", method="GET", level=LLAMA_STACK_API_V1)
    async def list_providers(self) -> ListProvidersResponse:
        """List providers.

        List all available providers.

        :returns: A ListProvidersResponse containing information about all providers.
        """
        ...

    @webmethod(route="/providers/{provider_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def inspect_provider(self, provider_id: str) -> ProviderInfo:
        """Get provider.

        Get detailed information about a specific provider.

        :param provider_id: The ID of the provider to inspect.
        :returns: A ProviderInfo object containing the provider's details.
        """
        ...
