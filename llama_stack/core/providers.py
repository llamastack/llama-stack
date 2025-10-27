# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from llama_stack.apis.providers import (
    ListProvidersResponse,
    ProviderInfo,
    Providers,
    RegisterProviderResponse,
    TestProviderConnectionResponse,
    UpdateProviderResponse,
)
from llama_stack.apis.providers.connection import (
    ProviderConnectionInfo,
    ProviderConnectionStatus,
    ProviderHealth,
)
from llama_stack.core.request_headers import get_authenticated_user
from llama_stack.core.resolver import ProviderWithSpec, instantiate_provider
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api, HealthResponse, HealthStatus

from .datatypes import StackRunConfig
from .utils.config import redact_sensitive_fields

logger = get_logger(name=__name__, category="core")

# Storage constants for dynamic provider connections
# Use composite key format: provider_connections:v1::{api}::{provider_id}
# This allows the same provider_id to be used for different APIs
PROVIDER_CONNECTIONS_PREFIX = "provider_connections:v1::"


class ProviderImplConfig(BaseModel):
    run_config: StackRunConfig
    provider_registry: Any | None = None  # ProviderRegistry from resolver
    dist_registry: Any | None = None  # DistributionRegistry
    policy: list[Any] | None = None  # list[AccessRule]


async def get_provider_impl(config, deps):
    impl = ProviderImpl(config, deps)
    await impl.initialize()
    return impl


class ProviderImpl(Providers):
    def __init__(self, config, deps):
        self.config = config
        self.deps = deps
        self.kvstore = None  # KVStore for dynamic provider persistence
        # Runtime cache uses composite key: "{api}::{provider_id}"
        # This allows the same provider_id to be used for different APIs
        self.dynamic_providers: dict[str, ProviderConnectionInfo] = {}  # Runtime cache
        self.dynamic_provider_impls: dict[str, Any] = {}  # Initialized provider instances

        # Store registry references for provider instantiation
        self.provider_registry = config.provider_registry
        self.dist_registry = config.dist_registry
        self.policy = config.policy or []

    async def initialize(self) -> None:
        # Initialize kvstore for dynamic providers
        # Use the metadata store from the new storage config structure
        if not (self.config.run_config.storage and self.config.run_config.storage.stores.metadata):
            raise RuntimeError(
                "No metadata store configured in storage.stores.metadata. "
                "Provider management requires a configured metadata store (kv_memory, kv_sqlite, etc)."
            )

        from llama_stack.providers.utils.kvstore import kvstore_impl

        self.kvstore = await kvstore_impl(self.config.run_config.storage.stores.metadata)
        logger.info("Initialized kvstore for dynamic provider management")

        # Load existing dynamic providers from kvstore
        await self._load_dynamic_providers()
        logger.info(f"Loaded {len(self.dynamic_providers)} existing dynamic providers from kvstore")

        for provider_id, conn_info in self.dynamic_providers.items():
            if conn_info.status == ProviderConnectionStatus.connected:
                try:
                    impl = await self._instantiate_provider(conn_info)
                    self.dynamic_provider_impls[provider_id] = impl
                except Exception as e:
                    logger.error(f"Failed to instantiate provider {provider_id}: {e}")
                    # Update status to failed
                    conn_info.status = ProviderConnectionStatus.failed
                    conn_info.error_message = str(e)
                    conn_info.updated_at = datetime.now(UTC)
                    await self._store_connection(conn_info)

    async def shutdown(self) -> None:
        logger.debug("ProviderImpl.shutdown")

        # Shutdown all dynamic provider instances
        for provider_id, impl in self.dynamic_provider_impls.items():
            try:
                if hasattr(impl, "shutdown"):
                    await impl.shutdown()
                    logger.debug(f"Shutdown dynamic provider {provider_id}")
            except Exception as e:
                logger.warning(f"Error shutting down dynamic provider {provider_id}: {e}")

        # Shutdown kvstore
        if self.kvstore and hasattr(self.kvstore, "shutdown"):
            await self.kvstore.shutdown()

    async def list_providers(self) -> ListProvidersResponse:
        run_config = self.config.run_config
        safe_config = StackRunConfig(**redact_sensitive_fields(run_config.model_dump()))
        providers_health = await self.get_providers_health()
        ret = []

        # Add static providers (from run.yaml)
        for api, providers in safe_config.providers.items():
            for p in providers:
                # Skip providers that are not enabled
                if p.provider_id is None:
                    continue
                ret.append(
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        config=p.config,
                        health=providers_health.get(api, {}).get(
                            p.provider_id,
                            HealthResponse(
                                status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                            ),
                        ),
                    )
                )

        # Add dynamic providers (from kvstore)
        for _provider_id, conn_info in self.dynamic_providers.items():
            # Redact sensitive config for API response
            redacted_config = self._redact_sensitive_config(conn_info.config)

            # Convert ProviderHealth to HealthResponse dict for API compatibility
            health_dict: HealthResponse | None = None
            if conn_info.health:
                health_dict = HealthResponse(
                    status=conn_info.health.status,
                    message=conn_info.health.message,
                )
                if conn_info.health.metrics:
                    health_dict["metrics"] = conn_info.health.metrics

            ret.append(
                ProviderInfo(
                    api=conn_info.api,
                    provider_id=conn_info.provider_id,
                    provider_type=conn_info.provider_type,
                    config=redacted_config,
                    health=health_dict
                    or HealthResponse(status=HealthStatus.NOT_IMPLEMENTED, message="No health check available"),
                )
            )

        return ListProvidersResponse(data=ret)

    async def inspect_provider(self, provider_id: str) -> ListProvidersResponse:
        """Get all providers with the given provider_id (deprecated).

        Returns all providers across all APIs that have this provider_id.
        This is deprecated - use inspect_provider_for_api() for unambiguous access.
        """
        all_providers = await self.list_providers()
        matching = [p for p in all_providers.data if p.provider_id == provider_id]

        if not matching:
            raise ValueError(f"Provider {provider_id} not found")

        return ListProvidersResponse(data=matching)

    async def list_providers_for_api(self, api: str) -> ListProvidersResponse:
        """List providers for a specific API."""
        all_providers = await self.list_providers()
        filtered = [p for p in all_providers.data if p.api == api]
        return ListProvidersResponse(data=filtered)

    async def inspect_provider_for_api(self, api: str, provider_id: str) -> ProviderInfo:
        """Get a specific provider for a specific API."""
        all_providers = await self.list_providers()
        for p in all_providers.data:
            if p.api == api and p.provider_id == provider_id:
                return p

        raise ValueError(f"Provider {provider_id} not found for API {api}")

    async def get_providers_health(self) -> dict[str, dict[str, HealthResponse]]:
        """Get health status for all providers.

        Returns:
            Dict[str, Dict[str, HealthResponse]]: A dictionary mapping API names to provider health statuses.
                Each API maps to a dictionary of provider IDs to their health responses.
        """
        providers_health: dict[str, dict[str, HealthResponse]] = {}

        # The timeout has to be long enough to allow all the providers to be checked, especially in
        # the case of the inference router health check since it checks all registered inference
        # providers.
        # The timeout must not be equal to the one set by health method for a given implementation,
        # otherwise we will miss some providers.
        timeout = 3.0

        async def check_provider_health(impl: Any) -> tuple[str, HealthResponse] | None:
            # Skip special implementations (inspect/providers) that don't have provider specs
            if not hasattr(impl, "__provider_spec__"):
                return None
            api_name = impl.__provider_spec__.api.name
            if not hasattr(impl, "health"):
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                    ),
                )

            try:
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                return api_name, health
            except TimeoutError:
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.ERROR, message=f"Health check timed out after {timeout} seconds"
                    ),
                )
            except Exception as e:
                return (
                    api_name,
                    HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"),
                )

        # Create tasks for all providers
        tasks = [check_provider_health(impl) for impl in self.deps.values()]

        # Wait for all health checks to complete
        results = await asyncio.gather(*tasks)

        # Organize results by API and provider ID
        for result in results:
            if result is None:  # Skip special implementations
                continue
            api_name, health_response = result
            providers_health[api_name] = health_response

        return providers_health

    # Storage helper methods for dynamic providers

    async def _store_connection(self, info: ProviderConnectionInfo) -> None:
        """Store provider connection info in kvstore.

        :param info: ProviderConnectionInfo to store
        """
        if not self.kvstore:
            raise RuntimeError("KVStore not initialized")

        # Use composite key: provider_connections:v1::{api}::{provider_id}
        key = f"{PROVIDER_CONNECTIONS_PREFIX}{info.api}::{info.provider_id}"
        await self.kvstore.set(key, info.model_dump_json())
        logger.debug(f"Stored provider connection: {info.api}::{info.provider_id}")

    async def _load_connection(self, provider_id: str) -> ProviderConnectionInfo | None:
        """Load provider connection info from kvstore.

        :param provider_id: Provider ID to load
        :returns: ProviderConnectionInfo if found, None otherwise
        """
        if not self.kvstore:
            return None

        key = f"{PROVIDER_CONNECTIONS_PREFIX}{provider_id}"
        value = await self.kvstore.get(key)
        if value:
            return ProviderConnectionInfo.model_validate_json(value)
        return None

    async def _delete_connection(self, api: str, provider_id: str) -> None:
        """Delete provider connection from kvstore.

        :param api: API namespace
        :param provider_id: Provider ID to delete
        """
        if not self.kvstore:
            raise RuntimeError("KVStore not initialized")

        # Use composite key: provider_connections:v1::{api}::{provider_id}
        key = f"{PROVIDER_CONNECTIONS_PREFIX}{api}::{provider_id}"
        await self.kvstore.delete(key)
        logger.debug(f"Deleted provider connection: {api}::{provider_id}")

    async def _list_connections(self) -> list[ProviderConnectionInfo]:
        """List all dynamic provider connections from kvstore.

        :returns: List of ProviderConnectionInfo
        """
        if not self.kvstore:
            return []

        start_key = PROVIDER_CONNECTIONS_PREFIX
        end_key = f"{PROVIDER_CONNECTIONS_PREFIX}\xff"
        values = await self.kvstore.values_in_range(start_key, end_key)
        return [ProviderConnectionInfo.model_validate_json(v) for v in values]

    async def _load_dynamic_providers(self) -> None:
        """Load dynamic providers from kvstore into runtime cache."""
        connections = await self._list_connections()
        for conn in connections:
            # Use composite key for runtime cache
            cache_key = f"{conn.api}::{conn.provider_id}"
            self.dynamic_providers[cache_key] = conn
            logger.debug(f"Loaded dynamic provider: {cache_key} (status: {conn.status})")

    def _find_provider_cache_key(self, provider_id: str) -> str | None:
        """Find the cache key for a provider by its provider_id.

        Since we use composite keys ({api}::{provider_id}), this searches for the matching key.
        Returns None if not found.
        """
        for key in self.dynamic_providers.keys():
            if key.endswith(f"::{provider_id}"):
                return key
        return None

    # Helper methods for dynamic provider management

    def _redact_sensitive_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive fields in provider config for API responses.

        :param config: Provider configuration dict
        :returns: Config with sensitive fields redacted
        """
        return redact_sensitive_fields(config)

    async def _instantiate_provider(self, conn_info: ProviderConnectionInfo) -> Any:
        """Instantiate a provider from connection info.

        Uses the resolver's instantiate_provider() to create a provider instance
        with all necessary dependencies.

        :param conn_info: Provider connection information
        :returns: Instantiated provider implementation
        :raises RuntimeError: If provider cannot be instantiated
        """
        if not self.provider_registry:
            raise RuntimeError("Provider registry not available for provider instantiation")
        if not self.dist_registry:
            raise RuntimeError("Distribution registry not available for provider instantiation")

        # Get provider spec from registry
        api = Api(conn_info.api)
        if api not in self.provider_registry:
            raise ValueError(f"API {conn_info.api} not found in provider registry")

        if conn_info.provider_type not in self.provider_registry[api]:
            raise ValueError(f"Provider type {conn_info.provider_type} not found for API {conn_info.api}")

        provider_spec = self.provider_registry[api][conn_info.provider_type]

        # Create ProviderWithSpec for instantiation
        provider_with_spec = ProviderWithSpec(
            provider_id=conn_info.provider_id,
            provider_type=conn_info.provider_type,
            config=conn_info.config,
            spec=provider_spec,
        )

        # Resolve dependencies
        deps = {}
        for dep_api in provider_spec.api_dependencies:
            if dep_api not in self.deps:
                raise RuntimeError(
                    f"Required dependency {dep_api.value} not available for provider {conn_info.provider_id}"
                )
            deps[dep_api] = self.deps[dep_api]

        # Add optional dependencies if available
        for dep_api in provider_spec.optional_api_dependencies:
            if dep_api in self.deps:
                deps[dep_api] = self.deps[dep_api]

        # Instantiate provider using resolver
        impl = await instantiate_provider(
            provider_with_spec,
            deps,
            {},  # inner_impls (empty for dynamic providers)
            self.dist_registry,
            self.config.run_config,
            self.policy,
        )

        logger.debug(f"Instantiated provider {conn_info.provider_id} (type={conn_info.provider_type})")
        return impl

    # Dynamic Provider Management Methods

    async def register_provider(
        self,
        api: str,
        provider_id: str,
        provider_type: str,
        config: dict[str, Any],
        attributes: dict[str, list[str]] | None = None,
    ) -> RegisterProviderResponse:
        """Register a new provider.

        This is used both for:
        - Providers from run.yaml (registered at startup)
        - Providers registered via API (registered at runtime)

        All providers are stored in kvstore and treated equally.
        """

        if not self.kvstore:
            raise RuntimeError("Dynamic provider management is not enabled (no kvstore configured)")

        # Use composite key to allow same provider_id for different APIs
        cache_key = f"{api}::{provider_id}"

        # Check if provider already exists for this API
        if cache_key in self.dynamic_providers:
            raise ValueError(f"Provider {provider_id} already exists for API {api}")

        # Get authenticated user as owner
        user = get_authenticated_user()

        # Create ProviderConnectionInfo
        now = datetime.now(UTC)
        conn_info = ProviderConnectionInfo(
            provider_id=provider_id,
            api=api,
            provider_type=provider_type,
            config=config,
            status=ProviderConnectionStatus.initializing,
            created_at=now,
            updated_at=now,
            owner=user,
            attributes=attributes,
        )

        try:
            # Store in kvstore
            await self._store_connection(conn_info)

            impl = await self._instantiate_provider(conn_info)
            # Use composite key for impl cache too
            self.dynamic_provider_impls[cache_key] = impl

            # Update status to connected after successful instantiation
            conn_info.status = ProviderConnectionStatus.connected
            conn_info.updated_at = datetime.now(UTC)

            logger.info(f"Registered and instantiated dynamic provider {provider_id} (api={api}, type={provider_type})")

            # Store updated status
            await self._store_connection(conn_info)

            # Add to runtime cache using composite key
            self.dynamic_providers[cache_key] = conn_info

            return RegisterProviderResponse(provider=conn_info)

        except Exception as e:
            # Mark as failed and store
            conn_info.status = ProviderConnectionStatus.failed
            conn_info.error_message = str(e)
            conn_info.updated_at = datetime.now(UTC)
            await self._store_connection(conn_info)
            self.dynamic_providers[cache_key] = conn_info

            logger.error(f"Failed to register provider {provider_id}: {e}")
            raise RuntimeError(f"Failed to register provider: {e}") from e

    async def update_provider(
        self,
        api: str,
        provider_id: str,
        config: dict[str, Any] | None = None,
        attributes: dict[str, list[str]] | None = None,
    ) -> UpdateProviderResponse:
        """Update an existing provider's configuration.

        Updates persist to kvstore and survive server restarts.
        This works for all providers (whether originally from run.yaml or API).
        """

        if not self.kvstore:
            raise RuntimeError("Dynamic provider management is not enabled (no kvstore configured)")

        # Use composite key
        cache_key = f"{api}::{provider_id}"
        if cache_key not in self.dynamic_providers:
            raise ValueError(f"Provider {provider_id} not found for API {api}")

        conn_info = self.dynamic_providers[cache_key]

        # Update config if provided
        if config is not None:
            conn_info.config.update(config)

        # Update attributes if provided
        if attributes is not None:
            conn_info.attributes = attributes

        conn_info.updated_at = datetime.now(UTC)
        conn_info.status = ProviderConnectionStatus.initializing

        try:
            # Store updated config
            await self._store_connection(conn_info)

            # Hot-reload: Shutdown old instance and reinstantiate with new config
            # Shutdown old instance if it exists
            if cache_key in self.dynamic_provider_impls:
                old_impl = self.dynamic_provider_impls[cache_key]
                if hasattr(old_impl, "shutdown"):
                    try:
                        await old_impl.shutdown()
                        logger.debug(f"Shutdown old instance of provider {provider_id}")
                    except Exception as e:
                        logger.warning(f"Error shutting down old instance of {provider_id}: {e}")

            # Reinstantiate with new config
            impl = await self._instantiate_provider(conn_info)
            self.dynamic_provider_impls[cache_key] = impl

            # Update status to connected after successful reinstantiation
            conn_info.status = ProviderConnectionStatus.connected
            conn_info.updated_at = datetime.now(UTC)
            await self._store_connection(conn_info)

            logger.info(f"Hot-reloaded dynamic provider {provider_id}")

            return UpdateProviderResponse(provider=conn_info)

        except Exception as e:
            conn_info.status = ProviderConnectionStatus.failed
            conn_info.error_message = str(e)
            conn_info.updated_at = datetime.now(UTC)
            await self._store_connection(conn_info)

            logger.error(f"Failed to update provider {provider_id}: {e}")
            raise RuntimeError(f"Failed to update provider: {e}") from e

    async def unregister_provider(self, api: str, provider_id: str) -> None:
        """Unregister a provider.

        Removes the provider from kvstore and shuts down its instance.
        This works for all providers (whether originally from run.yaml or API).
        """

        if not self.kvstore:
            raise RuntimeError("Dynamic provider management is not enabled (no kvstore configured)")

        # Use composite key
        cache_key = f"{api}::{provider_id}"
        if cache_key not in self.dynamic_providers:
            raise ValueError(f"Provider {provider_id} not found for API {api}")

        conn_info = self.dynamic_providers[cache_key]

        try:
            # Shutdown provider instance if it exists
            if cache_key in self.dynamic_provider_impls:
                impl = self.dynamic_provider_impls[cache_key]
                if hasattr(impl, "shutdown"):
                    await impl.shutdown()
                del self.dynamic_provider_impls[cache_key]

            # Remove from kvstore (using the api and provider_id from conn_info)
            await self._delete_connection(conn_info.api, provider_id)

            # Remove from runtime cache
            del self.dynamic_providers[cache_key]

            logger.info(f"Unregistered dynamic provider {provider_id}")

        except Exception as e:
            logger.error(f"Failed to unregister provider {provider_id}: {e}")
            raise RuntimeError(f"Failed to unregister provider: {e}") from e

    async def test_provider_connection(self, api: str, provider_id: str) -> TestProviderConnectionResponse:
        """Test a provider connection."""

        # Check if provider exists (static or dynamic)
        provider_impl = None
        cache_key = f"{api}::{provider_id}"

        # Check dynamic providers first (using composite keys)
        if cache_key in self.dynamic_provider_impls:
            provider_impl = self.dynamic_provider_impls[cache_key]

        # Check static providers
        if not provider_impl and provider_id in self.deps:
            provider_impl = self.deps[provider_id]

        if not provider_impl:
            return TestProviderConnectionResponse(
                success=False, error_message=f"Provider {provider_id} not found for API {api}"
            )

        # Check if provider has health method
        if not hasattr(provider_impl, "health"):
            return TestProviderConnectionResponse(
                success=False,
                health=HealthResponse(
                    status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                ),
            )

        # Call health check
        try:
            health_result = await asyncio.wait_for(provider_impl.health(), timeout=5.0)

            # Update health in dynamic provider cache if applicable
            if cache_key and cache_key in self.dynamic_providers:
                conn_info = self.dynamic_providers[cache_key]
                conn_info.health = ProviderHealth.from_health_response(health_result)
                conn_info.last_health_check = datetime.now(UTC)
                await self._store_connection(conn_info)

            logger.debug(f"Tested provider connection {provider_id}: status={health_result.get('status', 'UNKNOWN')}")

            return TestProviderConnectionResponse(
                success=health_result.get("status") == HealthStatus.OK,
                health=health_result,
            )

        except TimeoutError:
            health = HealthResponse(status=HealthStatus.ERROR, message="Health check timed out after 5 seconds")
            return TestProviderConnectionResponse(success=False, health=health)

        except Exception as e:
            health = HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")
            return TestProviderConnectionResponse(success=False, health=health, error_message=str(e))
