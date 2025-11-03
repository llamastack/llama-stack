# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import ListProvidersResponse, ProviderInfo


@runtime_checkable
class ProviderService(Protocol):
    """Providers

    Providers API for inspecting, listing, and modifying providers and their configurations.
    """

    async def list_providers(self) -> ListProvidersResponse:
        """List providers."""
        ...

    async def inspect_provider(self, provider_id: str) -> ProviderInfo:
        """Get provider."""
        ...
