# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Stack validation functions."""

from unittest.mock import AsyncMock

import pytest

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
    ConnectorType,
    ListConnectorsResponse,
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


class TestRegisterConnectors:
    """Tests for register_connectors function - config-driven CUD."""

    def _make_stack_config(self, connectors: list[ConnectorInput]) -> StackConfig:
        """Helper to create a StackConfig with given connectors."""
        return StackConfig(
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
            connectors=connectors,
        )

    async def test_register_connectors_creates_new(self):
        """Test that connectors in config are registered."""
        connectors_impl = AsyncMock()
        connectors_impl.list_connectors.return_value = ListConnectorsResponse(data=[])

        config = self._make_stack_config(
            [
                ConnectorInput(
                    connector_id="mcp-1",
                    connector_type=ConnectorType.MCP,
                    url="http://localhost:8080/mcp",
                    server_label="MCP Server 1",
                ),
                ConnectorInput(
                    connector_id="mcp-2",
                    connector_type=ConnectorType.MCP,
                    url="http://localhost:8081/mcp",
                ),
            ]
        )

        await register_connectors(config, {Api.connectors: connectors_impl})

        # Verify register_connector was called for each config connector
        assert connectors_impl.register_connector.call_count == 2

        # Check first connector registration
        call_args_1 = connectors_impl.register_connector.call_args_list[0]
        assert call_args_1.kwargs["connector_id"] == "mcp-1"
        assert call_args_1.kwargs["url"] == "http://localhost:8080/mcp"
        assert call_args_1.kwargs["server_label"] == "MCP Server 1"

        # Check second connector registration
        call_args_2 = connectors_impl.register_connector.call_args_list[1]
        assert call_args_2.kwargs["connector_id"] == "mcp-2"
        assert call_args_2.kwargs["url"] == "http://localhost:8081/mcp"

    async def test_register_connectors_removes_orphans(self):
        """Test that connectors not in config are removed (orphan cleanup)."""
        # Existing connectors in the store
        existing_connectors = [
            Connector(
                connector_id="keep-me",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8080/mcp",
            ),
            Connector(
                connector_id="orphan-1",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8081/mcp",
            ),
            Connector(
                connector_id="orphan-2",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8082/mcp",
            ),
        ]

        connectors_impl = AsyncMock()
        connectors_impl.list_connectors.return_value = ListConnectorsResponse(data=existing_connectors)

        # Config only has one connector
        config = self._make_stack_config(
            [
                ConnectorInput(
                    connector_id="keep-me",
                    connector_type=ConnectorType.MCP,
                    url="http://localhost:8080/mcp",
                ),
            ]
        )

        await register_connectors(config, {Api.connectors: connectors_impl})

        # Verify orphans were unregistered
        assert connectors_impl.unregister_connector.call_count == 2
        unregistered_ids = {call.args[0] for call in connectors_impl.unregister_connector.call_args_list}
        assert unregistered_ids == {"orphan-1", "orphan-2"}

    async def test_register_connectors_updates_existing(self):
        """Test that changed connectors in config are updated."""
        # Existing connector with old URL
        existing_connectors = [
            Connector(
                connector_id="my-mcp",
                connector_type=ConnectorType.MCP,
                url="http://old-host:8080/mcp",
                server_label="Old Label",
            ),
        ]

        connectors_impl = AsyncMock()
        connectors_impl.list_connectors.return_value = ListConnectorsResponse(data=existing_connectors)

        # Config has same connector with updated URL
        config = self._make_stack_config(
            [
                ConnectorInput(
                    connector_id="my-mcp",
                    connector_type=ConnectorType.MCP,
                    url="http://new-host:9090/mcp",
                    server_label="New Label",
                ),
            ]
        )

        await register_connectors(config, {Api.connectors: connectors_impl})

        # Verify register_connector was called with new values (upsert)
        connectors_impl.register_connector.assert_called_once_with(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://new-host:9090/mcp",
            server_label="New Label",
        )
        # No orphans to unregister
        connectors_impl.unregister_connector.assert_not_called()

    async def test_register_connectors_full_cud_flow(self):
        """Test complete Create/Update/Delete flow in a single operation."""
        # Existing connectors: one to keep (update), two orphans to delete
        existing_connectors = [
            Connector(
                connector_id="update-me",
                connector_type=ConnectorType.MCP,
                url="http://old-url:8080/mcp",
            ),
            Connector(
                connector_id="delete-me-1",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8081/mcp",
            ),
            Connector(
                connector_id="delete-me-2",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8082/mcp",
            ),
        ]

        connectors_impl = AsyncMock()
        connectors_impl.list_connectors.return_value = ListConnectorsResponse(data=existing_connectors)

        # Config: update one, create one new, two orphans will be deleted
        config = self._make_stack_config(
            [
                ConnectorInput(
                    connector_id="update-me",
                    connector_type=ConnectorType.MCP,
                    url="http://new-url:9090/mcp",
                ),
                ConnectorInput(
                    connector_id="create-me",
                    connector_type=ConnectorType.MCP,
                    url="http://localhost:8083/mcp",
                ),
            ]
        )

        await register_connectors(config, {Api.connectors: connectors_impl})

        # Verify Create/Update: 2 register calls
        assert connectors_impl.register_connector.call_count == 2
        registered_ids = {call.kwargs["connector_id"] for call in connectors_impl.register_connector.call_args_list}
        assert registered_ids == {"update-me", "create-me"}

        # Verify Delete: 2 orphans removed
        assert connectors_impl.unregister_connector.call_count == 2
        unregistered_ids = {call.args[0] for call in connectors_impl.unregister_connector.call_args_list}
        assert unregistered_ids == {"delete-me-1", "delete-me-2"}

    async def test_register_connectors_empty_config_removes_all(self):
        """Test that empty config removes all existing connectors."""
        existing_connectors = [
            Connector(
                connector_id="orphan-1",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8080/mcp",
            ),
            Connector(
                connector_id="orphan-2",
                connector_type=ConnectorType.MCP,
                url="http://localhost:8081/mcp",
            ),
        ]

        connectors_impl = AsyncMock()
        connectors_impl.list_connectors.return_value = ListConnectorsResponse(data=existing_connectors)

        # Empty config
        config = self._make_stack_config([])

        await register_connectors(config, {Api.connectors: connectors_impl})

        # No registrations
        connectors_impl.register_connector.assert_not_called()

        # All existing connectors should be unregistered
        assert connectors_impl.unregister_connector.call_count == 2

    async def test_register_connectors_skipped_if_api_not_available(self):
        """Test that registration is skipped if connectors API is not available."""
        config = self._make_stack_config(
            [
                ConnectorInput(
                    connector_id="mcp-1",
                    connector_type=ConnectorType.MCP,
                    url="http://localhost:8080/mcp",
                ),
            ]
        )

        # No connectors API in impls
        await register_connectors(config, {})

        # Should complete without error (early return)
