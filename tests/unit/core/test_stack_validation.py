# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Stack validation functions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.core.connectors.connectors import KEY_PREFIX, ConnectorServiceImpl
from llama_stack.core.datatypes import (
    QualifiedModel,
    RewriteQueryParams,
    SafetyConfig,
    StackConfig,
    VectorStoresConfig,
)
from llama_stack.core.stack import register_connectors, validate_safety_config, validate_vector_stores_config
from llama_stack.core.storage.datatypes import ServerStoresConfig, StorageConfig
from llama_stack_api import (
    Api,
    Connector,
    ConnectorInput,
    ConnectorSource,
    ConnectorType,
    ListModelsResponse,
    ListShieldsResponse,
    Model,
    ModelType,
    Shield,
)


class TestVectorStoresValidation:
    async def test_validate_missing_model(self):
        """Test validation fails when model not found."""
        run_config = StackConfig(
            distro_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
            vector_stores=VectorStoresConfig(
                default_provider_id="faiss",
                default_embedding_model=QualifiedModel(
                    provider_id="p",
                    model_id="missing",
                ),
            ),
        )
        mock_models = AsyncMock()
        mock_models.list_models.return_value = ListModelsResponse(data=[])

        with pytest.raises(ValueError, match="not found"):
            await validate_vector_stores_config(run_config.vector_stores, {Api.models: mock_models})

    async def test_validate_success(self):
        """Test validation passes with valid model."""
        run_config = StackConfig(
            distro_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
            vector_stores=VectorStoresConfig(
                default_provider_id="faiss",
                default_embedding_model=QualifiedModel(
                    provider_id="p",
                    model_id="valid",
                ),
            ),
        )
        mock_models = AsyncMock()
        mock_models.list_models.return_value = ListModelsResponse(
            data=[
                Model(
                    identifier="p/valid",  # Must match provider_id/model_id format
                    model_type=ModelType.embedding,
                    metadata={"embedding_dimension": 768},
                    provider_id="p",
                    provider_resource_id="valid",
                )
            ]
        )

        await validate_vector_stores_config(run_config.vector_stores, {Api.models: mock_models})

    async def test_validate_rewrite_query_prompt_missing_placeholder(self):
        """Test validation fails when prompt template is missing {query} placeholder."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=r"prompt must contain \{query\} placeholder"):
            RewriteQueryParams(
                prompt="This prompt has no placeholder",
            )


class TestSafetyConfigValidation:
    async def test_validate_success(self):
        safety_config = SafetyConfig(default_shield_id="shield-1")

        shield = Shield(
            identifier="shield-1",
            provider_id="provider-x",
            provider_resource_id="model-x",
            params={},
        )

        shields_impl = AsyncMock()
        shields_impl.list_shields.return_value = ListShieldsResponse(data=[shield])

        await validate_safety_config(safety_config, {Api.shields: shields_impl, Api.safety: AsyncMock()})

    async def test_validate_wrong_shield_id(self):
        safety_config = SafetyConfig(default_shield_id="wrong-shield-id")

        shields_impl = AsyncMock()
        shields_impl.list_shields.return_value = ListShieldsResponse(
            data=[
                Shield(
                    identifier="shield-1",
                    provider_resource_id="model-x",
                    provider_id="provider-x",
                    params={},
                )
            ]
        )
        with pytest.raises(ValueError, match="wrong-shield-id"):
            await validate_safety_config(safety_config, {Api.shields: shields_impl, Api.safety: AsyncMock()})


class TestConnectorConfigSync:
    """Tests for connector config synchronization during stack startup."""

    @pytest.fixture
    def mock_kvstore(self):
        """Create a mock KVStore with in-memory storage."""
        storage = {}

        class MockKVStore:
            async def set(self, key, value):
                storage[key] = value

            async def get(self, key):
                return storage.get(key)

            async def delete(self, key):
                del storage[key]

            async def keys_in_range(self, start, end):
                return [k for k in storage.keys() if start <= k < end]

            async def close(self):
                pass

            @property
            def _storage(self):
                return storage

        return MockKVStore()

    @pytest.fixture
    async def connector_service(self, mock_kvstore):
        """Create a ConnectorServiceImpl with mocked dependencies."""
        mock_config = MagicMock()

        with patch(
            "llama_stack.core.connectors.connectors.kvstore_impl",
            return_value=mock_kvstore,
        ):
            service = ConnectorServiceImpl(mock_config)
            service.kvstore = mock_kvstore
            return service

    async def test_sync_config_connectors_with_drift(self, connector_service, mock_kvstore):
        """Test that stale config connectors are removed and new ones are added during sync.

        Simulates server restart where the config has changed:
        - Stale connectors in KV store that are no longer in config should be removed
        - New connectors in config that aren't in KV store should be added
        - Existing connectors with different values should be updated
        - Existing connectors that are not being updated should be left alone
        - Final KV store state should match the new config exactly
        """
        # === Setup: Pre-populate KV store with "old" config connectors (simulating previous run) ===
        stale_connector_1 = Connector(
            connector_id="mcp-1",
            connector_type=ConnectorType.MCP,
            url="http://old-server-1:8080/mcp",
            source=ConnectorSource.config,
        )
        stale_connector_2 = Connector(
            connector_id="stale-mcp-2",
            connector_type=ConnectorType.MCP,
            url="http://old-server-2:8080/mcp",
            source=ConnectorSource.config,
        )
        # Also add an API-registered connector that should NOT be removed
        api_connector = Connector(
            connector_id="mcp-4",
            connector_type=ConnectorType.MCP,
            url="http://api-server:8080/mcp",
            source=ConnectorSource.api,
        )

        api_connector_2 = Connector(
            connector_id="api-registered",
            connector_type=ConnectorType.MCP,
            url="http://api-server-2:8080/mcp",
            source=ConnectorSource.api,
        )

        # Manually insert into KV store (simulating previous server state)
        for conn in [stale_connector_1, stale_connector_2, api_connector, api_connector_2]:
            payload = conn.model_dump()
            payload["source"] = conn.source
            await mock_kvstore.set(f"{KEY_PREFIX}{conn.connector_id}", json.dumps(payload))

        # Verify initial state: 3 connectors in KV store
        initial_keys = await mock_kvstore.keys_in_range(KEY_PREFIX, KEY_PREFIX + "\uffff")
        assert len(initial_keys) == 4

        # === Define "new" config connectors ===
        new_config_connectors = [
            ConnectorInput(connector_id="mcp-1", url="http://new-server-1:8080/mcp"),
            ConnectorInput(connector_id="new-mcp-2", url="http://new-server-2:8080/mcp"),
            ConnectorInput(connector_id="new-mcp-3", url="http://new-server-3:8080/mcp"),
            ConnectorInput(connector_id="mcp-4", url="http://new-server-4:8080/mcp"),
        ]

        # === Sync operation via register_connectors (as would happen during server startup) ===
        mock_run_config = MagicMock()
        mock_run_config.connectors = new_config_connectors

        impls = {Api.connectors: connector_service}

        await register_connectors(mock_run_config, impls)

        # === Verify final state ===
        final_connectors = await connector_service.list_connectors()

        assert len(final_connectors.data) == 5

        final_ids = {c.connector_id for c in final_connectors.data}
        expected_ids = {"mcp-1", "new-mcp-2", "new-mcp-3", "mcp-4", "api-registered"}
        assert final_ids == expected_ids

        # Verify stale connectors are gone
        assert "stale-mcp-2" not in final_ids

        # Build a lookup for easy verification
        connectors_by_id = {c.connector_id: c for c in final_connectors.data}

        # Verify mcp-1: was updated from old URL to new URL, source should be config
        assert connectors_by_id["mcp-1"].url == "http://new-server-1:8080/mcp"
        assert connectors_by_id["mcp-1"].source == ConnectorSource.config

        # Verify new-mcp-2: newly added, should have config source
        assert connectors_by_id["new-mcp-2"].url == "http://new-server-2:8080/mcp"
        assert connectors_by_id["new-mcp-2"].source == ConnectorSource.config

        # Verify new-mcp-3: newly added, should have config source
        assert connectors_by_id["new-mcp-3"].url == "http://new-server-3:8080/mcp"
        assert connectors_by_id["new-mcp-3"].source == ConnectorSource.config

        # Verify mcp-4: was API-registered, now overwritten by config with new URL
        assert connectors_by_id["mcp-4"].url == "http://new-server-4:8080/mcp"
        assert connectors_by_id["mcp-4"].source == ConnectorSource.config

        # Verify api-registered: untouched API connector, should retain original values
        assert connectors_by_id["api-registered"].url == "http://api-server-2:8080/mcp"
        assert connectors_by_id["api-registered"].source == ConnectorSource.api
