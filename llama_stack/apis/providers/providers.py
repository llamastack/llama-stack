# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.providers.connection import ProviderConnectionInfo
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.providers.datatypes import HealthResponse
from llama_stack.schema_utils import json_schema_type, webmethod


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


class ListProvidersResponse(BaseModel):
    """Response containing a list of all available providers.

    :param data: List of provider information objects
    """

    data: list[ProviderInfo]


# ===== Dynamic Provider Management API Models =====


@json_schema_type
class RegisterProviderRequest(BaseModel):
    """Request to register a new dynamic provider.

    :param provider_id: Unique identifier for the provider instance
    :param api: API namespace (e.g., 'inference', 'vector_io', 'safety')
    :param provider_type: Provider type identifier (e.g., 'remote::openai', 'inline::faiss')
    :param config: Provider-specific configuration (API keys, endpoints, etc.)
    :param attributes: Optional key-value attributes for ABAC access control
    """

    provider_id: str
    api: str
    provider_type: str
    config: dict[str, Any]
    attributes: dict[str, list[str]] | None = None


@json_schema_type
class RegisterProviderResponse(BaseModel):
    """Response after registering a provider.

    :param provider: Information about the registered provider
    """

    provider: ProviderConnectionInfo


@json_schema_type
class UpdateProviderRequest(BaseModel):
    """Request to update an existing provider's configuration.

    :param config: New configuration parameters (will be merged with existing)
    :param attributes: Optional updated attributes for access control
    """

    config: dict[str, Any] | None = None
    attributes: dict[str, list[str]] | None = None


@json_schema_type
class UpdateProviderResponse(BaseModel):
    """Response after updating a provider.

    :param provider: Updated provider information
    """

    provider: ProviderConnectionInfo


@json_schema_type
class UnregisterProviderResponse(BaseModel):
    """Response after unregistering a provider.

    :param success: Whether the operation succeeded
    :param message: Optional status message
    """

    success: bool
    message: str | None = None


@json_schema_type
class TestProviderConnectionResponse(BaseModel):
    """Response from testing a provider connection.

    :param success: Whether the connection test succeeded
    :param health: Health status from the provider
    :param error_message: Error message if test failed
    """

    success: bool
    health: HealthResponse | None = None
    error_message: str | None = None


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

    # ===== Dynamic Provider Management Methods =====

    @webmethod(route="/admin/providers", method="POST", level=LLAMA_STACK_API_V1)
    async def register_provider(
        self,
        provider_id: str,
        api: str,
        provider_type: str,
        config: dict[str, Any],
        attributes: dict[str, list[str]] | None = None,
    ) -> RegisterProviderResponse:
        """Register a new dynamic provider.

        Register a new provider instance at runtime. The provider will be validated,
        instantiated, and persisted to the kvstore. Requires appropriate ABAC permissions.

        :param provider_id: Unique identifier for this provider instance.
        :param api: API namespace this provider implements.
        :param provider_type: Provider type (e.g., 'remote::openai').
        :param config: Provider configuration (API keys, endpoints, etc.).
        :param attributes: Optional attributes for ABAC access control.
        :returns: RegisterProviderResponse with the registered provider info.
        """
        ...

    @webmethod(route="/admin/providers/{provider_id}", method="PUT", level=LLAMA_STACK_API_V1)
    async def update_provider(
        self,
        provider_id: str,
        config: dict[str, Any] | None = None,
        attributes: dict[str, list[str]] | None = None,
    ) -> UpdateProviderResponse:
        """Update an existing provider's configuration.

        Update the configuration and/or attributes of a dynamic provider. The provider
        will be re-instantiated with the new configuration (hot-reload). Static providers
        from run.yaml cannot be updated.

        :param provider_id: ID of the provider to update
        :param config: New configuration parameters (merged with existing)
        :param attributes: New attributes for access control
        :returns: UpdateProviderResponse with updated provider info
        """
        ...

    @webmethod(route="/admin/providers/{provider_id}", method="DELETE", level=LLAMA_STACK_API_V1)
    async def unregister_provider(self, provider_id: str) -> None:
        """Unregister a dynamic provider.

        Remove a dynamic provider, shutting down its instance and removing it from
        the kvstore. Static providers from run.yaml cannot be unregistered.

        :param provider_id: ID of the provider to unregister.
        """
        ...

    @webmethod(route="/providers/{provider_id}/test", method="POST", level=LLAMA_STACK_API_V1)
    async def test_provider_connection(self, provider_id: str) -> TestProviderConnectionResponse:
        """Test a provider connection.

        Execute a health check on a provider to verify it is reachable and functioning.
        Works for both static and dynamic providers.

        :param provider_id: ID of the provider to test.
        :returns: TestProviderConnectionResponse with health status.
        """
        ...
