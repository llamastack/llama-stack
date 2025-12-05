# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

# Import types from admin module to avoid duplication
from llama_stack_api.admin.models import ListProvidersResponse, ProviderInfo
from llama_stack_api.schema_utils import webmethod
from llama_stack_api.version import LLAMA_STACK_API_V1


@runtime_checkable
class Providers(Protocol):
    """Providers

    Providers API for inspecting, listing, and modifying providers and their configurations.
    """

    @webmethod(route="/providers", method="GET", level=LLAMA_STACK_API_V1, deprecated=True)
    async def list_providers(self) -> ListProvidersResponse:
        """List providers.

        List all available providers.

        :returns: A ListProvidersResponse containing information about all providers.
        """
        ...

    @webmethod(route="/providers/{provider_id}", method="GET", level=LLAMA_STACK_API_V1, deprecated=True)
    async def inspect_provider(self, provider_id: str) -> ProviderInfo:
        """Get provider.

        Get detailed information about a specific provider.

        :param provider_id: The ID of the provider to inspect.
        :returns: A ProviderInfo object containing the provider's details.
        """
        ...
