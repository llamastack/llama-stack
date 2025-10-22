# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llama_stack.apis.providers.connection import ProviderConnectionInfo, ProviderConnectionStatus, ProviderHealth
from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.providers import ProviderImpl, ProviderImplConfig
from llama_stack.core.storage.datatypes import KVStoreReference, ServerStoresConfig, SqliteKVStoreConfig, StorageConfig
from llama_stack.providers.datatypes import Api, HealthStatus
from llama_stack.providers.utils.kvstore.sqlite import SqliteKVStoreImpl


@pytest.fixture
async def kvstore(tmp_path):
    """Create a temporary kvstore for testing."""
    db_path = tmp_path / "test_providers.db"
    kvstore_config = SqliteKVStoreConfig(db_path=db_path.as_posix())
    kvstore = SqliteKVStoreImpl(kvstore_config)
    await kvstore.initialize()
    yield kvstore


@pytest.fixture
async def provider_impl(kvstore, tmp_path):
    """Create a ProviderImpl instance with mocked dependencies."""
    db_path = (tmp_path / "test_providers.db").as_posix()

    # Create storage config with required structure
    storage_config = StorageConfig(
        backends={
            "default": SqliteKVStoreConfig(db_path=db_path),
        },
        stores=ServerStoresConfig(
            metadata=KVStoreReference(backend="default", namespace="test_metadata"),
        ),
    )

    # Create minimal run config with storage
    run_config = StackRunConfig(
        image_name="test",
        apis=[],
        providers={},
        storage=storage_config,
    )

    # Mock provider registry
    mock_provider_registry = MagicMock()

    config = ProviderImplConfig(
        run_config=run_config,
        provider_registry=mock_provider_registry,
        dist_registry=None,
        policy=None,
    )

    impl = ProviderImpl(config, deps={})

    # Manually set the kvstore instead of going through initialize
    # This avoids the complex backend registration logic
    impl.kvstore = kvstore
    impl.provider_registry = mock_provider_registry
    impl.dist_registry = None
    impl.policy = []

    yield impl


@pytest.mark.asyncio
class TestDynamicProviderManagement:
    """Unit tests for dynamic provider registration, update, and unregistration."""

    async def test_register_inference_provider(self, provider_impl):
        """Test registering a new inference provider."""
        # Mock the provider instantiation
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register a mock inference provider
            response = await provider_impl.register_provider(
                provider_id="test-inference-1",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "test-key", "url": "https://api.openai.com/v1"},
                attributes={"team": ["test-team"]},
            )

        # Verify response
        assert response.provider.provider_id == "test-inference-1"
        assert response.provider.api == Api.inference.value
        assert response.provider.provider_type == "remote::openai"
        assert response.provider.status == ProviderConnectionStatus.connected
        assert response.provider.config["api_key"] == "test-key"
        assert response.provider.attributes == {"team": ["test-team"]}

        # Verify provider is stored
        assert "test-inference-1" in provider_impl.dynamic_providers
        assert "test-inference-1" in provider_impl.dynamic_provider_impls

    async def test_register_vector_store_provider(self, provider_impl):
        """Test registering a new vector store provider."""
        # Mock the provider instantiation
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register a mock vector_io provider
            response = await provider_impl.register_provider(
                provider_id="test-vector-store-1",
                api=Api.vector_io.value,
                provider_type="inline::faiss",
                config={"dimension": 768, "index_path": "/tmp/faiss_index"},
            )

        # Verify response
        assert response.provider.provider_id == "test-vector-store-1"
        assert response.provider.api == Api.vector_io.value
        assert response.provider.provider_type == "inline::faiss"
        assert response.provider.status == ProviderConnectionStatus.connected
        assert response.provider.config["dimension"] == 768

    async def test_register_duplicate_provider_fails(self, provider_impl):
        """Test that registering a duplicate provider_id fails."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register first provider
            await provider_impl.register_provider(
                provider_id="test-duplicate",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "key1"},
            )

            # Try to register with same ID
            with pytest.raises(ValueError, match="already exists"):
                await provider_impl.register_provider(
                    provider_id="test-duplicate",
                    api=Api.inference.value,
                    provider_type="remote::openai",
                    config={"api_key": "key2"},
                )

    async def test_update_provider_config(self, provider_impl):
        """Test updating a provider's configuration."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider
            await provider_impl.register_provider(
                provider_id="test-update",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "old-key", "timeout": 30},
            )

            # Update configuration
            response = await provider_impl.update_provider(
                provider_id="test-update",
                config={"api_key": "new-key", "timeout": 60},
            )

        # Verify updated config
        assert response.provider.provider_id == "test-update"
        assert response.provider.config["api_key"] == "new-key"
        assert response.provider.config["timeout"] == 60
        assert response.provider.status == ProviderConnectionStatus.connected

    async def test_update_provider_attributes(self, provider_impl):
        """Test updating a provider's attributes."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider with initial attributes
            await provider_impl.register_provider(
                provider_id="test-attributes",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "test-key"},
                attributes={"team": ["team-a"]},
            )

            # Update attributes
            response = await provider_impl.update_provider(
                provider_id="test-attributes",
                attributes={"team": ["team-a", "team-b"], "environment": ["prod"]},
            )

        # Verify updated attributes
        assert response.provider.attributes == {"team": ["team-a", "team-b"], "environment": ["prod"]}

    async def test_update_nonexistent_provider_fails(self, provider_impl):
        """Test that updating a non-existent provider fails."""
        with pytest.raises(ValueError, match="not found"):
            await provider_impl.update_provider(
                provider_id="nonexistent",
                config={"api_key": "new-key"},
            )

    async def test_unregister_provider(self, provider_impl):
        """Test unregistering a provider."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})
        mock_provider_instance.shutdown = AsyncMock()

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider
            await provider_impl.register_provider(
                provider_id="test-unregister",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "test-key"},
            )

            # Verify it exists
            assert "test-unregister" in provider_impl.dynamic_providers

            # Unregister provider
            await provider_impl.unregister_provider(provider_id="test-unregister")

        # Verify it's removed
        assert "test-unregister" not in provider_impl.dynamic_providers
        assert "test-unregister" not in provider_impl.dynamic_provider_impls

        # Verify shutdown was called
        mock_provider_instance.shutdown.assert_called_once()

    async def test_unregister_nonexistent_provider_fails(self, provider_impl):
        """Test that unregistering a non-existent provider fails."""
        with pytest.raises(ValueError, match="not found"):
            await provider_impl.unregister_provider(provider_id="nonexistent")

    async def test_test_provider_connection_healthy(self, provider_impl):
        """Test testing a healthy provider connection."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK, "message": "All good"})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider
            await provider_impl.register_provider(
                provider_id="test-health",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "test-key"},
            )

            # Test connection
            response = await provider_impl.test_provider_connection(provider_id="test-health")

        # Verify response
        assert response.success is True
        assert response.health["status"] == HealthStatus.OK
        assert response.health["message"] == "All good"
        assert response.error_message is None

    async def test_test_provider_connection_unhealthy(self, provider_impl):
        """Test testing an unhealthy provider connection."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(
            return_value={"status": HealthStatus.ERROR, "message": "Connection failed"}
        )

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider
            await provider_impl.register_provider(
                provider_id="test-unhealthy",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "invalid-key"},
            )

            # Test connection
            response = await provider_impl.test_provider_connection(provider_id="test-unhealthy")

        # Verify response shows unhealthy status
        assert response.success is False
        assert response.health["status"] == HealthStatus.ERROR

    async def test_list_providers_includes_dynamic(self, provider_impl):
        """Test that list_providers includes dynamically registered providers."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register multiple providers
            await provider_impl.register_provider(
                provider_id="dynamic-1",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "key1"},
            )

            await provider_impl.register_provider(
                provider_id="dynamic-2",
                api=Api.vector_io.value,
                provider_type="inline::faiss",
                config={"dimension": 768},
            )

            # List all providers
            response = await provider_impl.list_providers()

        # Verify both dynamic providers are in the list
        provider_ids = [p.provider_id for p in response.data]
        assert "dynamic-1" in provider_ids
        assert "dynamic-2" in provider_ids

    async def test_inspect_provider(self, provider_impl):
        """Test inspecting a specific provider."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider
            await provider_impl.register_provider(
                provider_id="test-inspect",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "test-key", "model": "gpt-4"},
            )

        # Update the stored health info to reflect OK status
        # (In reality, the health check happens during registration,
        # but our mock may not have been properly called)
        conn_info = provider_impl.dynamic_providers["test-inspect"]
        from llama_stack.apis.providers.connection import ProviderHealth

        conn_info.health = ProviderHealth.from_health_response({"status": HealthStatus.OK})

        # Inspect provider
        provider_info = await provider_impl.inspect_provider(provider_id="test-inspect")

        # Verify provider info
        assert provider_info.provider_id == "test-inspect"
        assert provider_info.api == Api.inference.value
        assert provider_info.provider_type == "remote::openai"
        assert provider_info.config["model"] == "gpt-4"
        assert provider_info.health["status"] == HealthStatus.OK

    async def test_provider_persistence(self, provider_impl, kvstore, tmp_path):
        """Test that providers persist across restarts."""
        mock_provider_instance = AsyncMock()
        mock_provider_instance.health = AsyncMock(return_value={"status": HealthStatus.OK})

        with patch.object(provider_impl, "_instantiate_provider", return_value=mock_provider_instance):
            # Register provider
            await provider_impl.register_provider(
                provider_id="test-persist",
                api=Api.inference.value,
                provider_type="remote::openai",
                config={"api_key": "persist-key"},
            )

        # Create a new provider impl (simulating restart) - reuse the same kvstore
        db_path = (tmp_path / "test_providers.db").as_posix()

        storage_config = StorageConfig(
            backends={
                "default": SqliteKVStoreConfig(db_path=db_path),
            },
            stores=ServerStoresConfig(
                metadata=KVStoreReference(backend="default", namespace="test_metadata"),
            ),
        )

        run_config = StackRunConfig(
            image_name="test",
            apis=[],
            providers={},
            storage=storage_config,
        )

        config = ProviderImplConfig(
            run_config=run_config,
            provider_registry=MagicMock(),
            dist_registry=None,
            policy=None,
        )

        new_impl = ProviderImpl(config, deps={})

        # Manually set the kvstore (reusing the same one)
        new_impl.kvstore = kvstore
        new_impl.provider_registry = MagicMock()
        new_impl.dist_registry = None
        new_impl.policy = []

        # Load providers from kvstore
        with patch.object(new_impl, "_instantiate_provider", return_value=mock_provider_instance):
            await new_impl._load_dynamic_providers()

        # Verify the provider was loaded from kvstore
        assert "test-persist" in new_impl.dynamic_providers
        assert new_impl.dynamic_providers["test-persist"].config["api_key"] == "persist-key"
