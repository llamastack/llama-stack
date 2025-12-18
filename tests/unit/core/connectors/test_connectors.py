# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Connectors API implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.core.connectors.connectors import (
    KEY_PREFIX,
    ConnectorServiceImpl,
)
from llama_stack_api import (
    Connector,
    ConnectorNotFoundError,
    ConnectorSource,
    ConnectorToolNotFoundError,
    ConnectorType,
    ListConnectorsResponse,
    ListToolsResponse,
    OpenAIResponseInputToolMCP,
    ToolDef,
)

# --- Fixtures ---


@pytest.fixture
def mock_kvstore():
    """Create a mock KVStore with in-memory storage."""
    storage = {}

    class MockKVStore:
        async def set(self, key, value):
            storage[key] = value

        async def get(self, key):
            return storage.get(key)

        async def keys_in_range(self, start, end):
            return [k for k in storage.keys() if start <= k < end]

        async def close(self):
            pass

        @property
        def _storage(self):
            return storage

    return MockKVStore()


@pytest.fixture
async def connector_service(mock_kvstore):
    """Create a ConnectorServiceImpl with mocked dependencies."""
    # Create a minimal mock config - we'll inject the kvstore directly
    mock_config = MagicMock()

    with patch(
        "llama_stack.core.connectors.connectors.kvstore_impl",
        return_value=mock_kvstore,
    ):
        service = ConnectorServiceImpl(mock_config)
        service.kvstore = mock_kvstore  # Inject directly
        return service


@pytest.fixture
def sample_tool_def():
    """Create a sample ToolDef for testing."""
    return ToolDef(
        name="get_weather",
        description="Get weather for a location",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        output_schema={"type": "object"},
    )


@pytest.fixture
def mock_connectors_api():
    """Create a mock connectors API."""
    api = AsyncMock()
    return api


@pytest.fixture
def sample_connector():
    """Create a sample connector."""
    return Connector(
        connector_id="my-mcp-server",
        connector_type=ConnectorType.MCP,
        url="http://localhost:8080/mcp",
        server_label="My MCP Server",
        server_name="Test Server",
        source=ConnectorSource.config,
    )


# --- register_connector tests ---


class TestRegisterConnector:
    """Tests for register_connector method."""

    async def test_register_new_connector(self, connector_service, mock_kvstore):
        """Test registering a new connector creates it in the store."""
        result = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            server_label="My MCP",
            source=ConnectorSource.config,
        )

        assert result.connector_id == "my-mcp"
        assert result.connector_type == ConnectorType.MCP
        assert result.url == "http://localhost:8080/mcp"
        assert result.server_label == "My MCP"

        # Verify stored in kvstore
        stored = await mock_kvstore.get(f"{KEY_PREFIX}my-mcp")
        assert stored is not None

    async def test_register_connector_idempotent(self, connector_service, mock_kvstore):
        """Test that re-registering a connector updates it (idempotent)."""
        # Register first time
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            server_label="Original Label",
            source=ConnectorSource.config,
        )

        # Register again with different label
        result = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            server_label="Updated Label",
            source=ConnectorSource.config,
        )

        assert result.server_label == "Updated Label"

        # Should only have one entry
        keys = await mock_kvstore.keys_in_range(KEY_PREFIX, KEY_PREFIX + "\uffff")
        assert len(keys) == 1


# --- list_connectors tests ---


class TestListConnectors:
    """Tests for list_connectors method."""

    async def test_list_connectors_empty(self, connector_service):
        """Test listing connectors when none registered."""
        result = await connector_service.list_connectors()

        assert isinstance(result, ListConnectorsResponse)
        assert result.data == []

    async def test_list_connectors_returns_all(self, connector_service):
        """Test listing returns all registered connectors."""
        # Register multiple connectors
        await connector_service.register_connector(
            connector_id="mcp-1",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8081/mcp",
            source=ConnectorSource.config,
        )
        await connector_service.register_connector(
            connector_id="mcp-2",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8082/mcp",
            source=ConnectorSource.config,
        )

        result = await connector_service.list_connectors()

        assert len(result.data) == 2
        connector_ids = {c.connector_id for c in result.data}
        assert connector_ids == {"mcp-1", "mcp-2"}


# --- get_connector tests ---


class TestGetConnector:
    """Tests for get_connector method."""

    async def test_get_connector_not_found(self, connector_service):
        """Test getting a non-existent connector raises error."""
        with pytest.raises(ConnectorNotFoundError) as exc_info:
            await connector_service.get_connector("non-existent")

        assert "non-existent" in str(exc_info.value)

    async def test_get_connector_returns_with_server_info(self, connector_service):
        """Test getting a connector fetches MCP server info."""
        # Register a connector
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            source=ConnectorSource.config,
        )

        # Mock the MCP server info response
        mock_server_info = MagicMock()
        mock_server_info.name = "Test MCP Server"
        mock_server_info.description = "A test server"
        mock_server_info.version = "1.0.0"

        with patch("llama_stack.core.connectors.connectors._import_mcp_tools") as mock_import:
            mock_get_info = AsyncMock(return_value=mock_server_info)
            mock_import.return_value = (mock_get_info, None)

            result = await connector_service.get_connector("my-mcp")

        assert result.connector_id == "my-mcp"
        assert result.server_name == "Test MCP Server"
        assert result.server_description == "A test server"
        assert result.server_version == "1.0.0"

    async def test_get_connector_with_authorization(self, connector_service):
        """Test that authorization is passed to MCP server."""
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            source=ConnectorSource.config,
        )

        mock_server_info = MagicMock()
        mock_server_info.name = "Server"
        mock_server_info.description = None
        mock_server_info.version = None

        with patch("llama_stack.core.connectors.connectors._import_mcp_tools") as mock_import:
            mock_get_info = AsyncMock(return_value=mock_server_info)
            mock_import.return_value = (mock_get_info, None)

            await connector_service.get_connector("my-mcp", authorization="Bearer token123")

            mock_get_info.assert_called_once_with(
                "http://localhost:8080/mcp",
                authorization="Bearer token123",
            )


# --- list_connector_tools tests ---


class TestListConnectorTools:
    """Tests for list_connector_tools method."""

    async def test_list_connector_tools_returns_tools(self, connector_service, sample_tool_def):
        """Test listing tools from a connector."""
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            source=ConnectorSource.config,
        )

        mock_server_info = MagicMock()
        mock_server_info.name = "Server"
        mock_server_info.description = None
        mock_server_info.version = None

        mock_tools_response = MagicMock()
        mock_tools_response.data = [sample_tool_def]

        with patch("llama_stack.core.connectors.connectors._import_mcp_tools") as mock_import:
            mock_get_info = AsyncMock(return_value=mock_server_info)
            mock_list_tools = AsyncMock(return_value=mock_tools_response)
            mock_import.return_value = (mock_get_info, mock_list_tools)

            result = await connector_service.list_connector_tools("my-mcp")

        assert isinstance(result, ListToolsResponse)
        assert len(result.data) == 1
        assert result.data[0].name == "get_weather"


# --- get_connector_tool tests ---


class TestGetConnectorTool:
    """Tests for get_connector_tool method."""

    async def test_get_connector_tool_found(self, connector_service, sample_tool_def):
        """Test getting a specific tool by name."""
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            source=ConnectorSource.config,
        )

        mock_server_info = MagicMock()
        mock_server_info.name = "Server"
        mock_server_info.description = None
        mock_server_info.version = None

        mock_tools_response = MagicMock()
        mock_tools_response.data = [sample_tool_def]

        with patch("llama_stack.core.connectors.connectors._import_mcp_tools") as mock_import:
            mock_get_info = AsyncMock(return_value=mock_server_info)
            mock_list_tools = AsyncMock(return_value=mock_tools_response)
            mock_import.return_value = (mock_get_info, mock_list_tools)

            result = await connector_service.get_connector_tool("my-mcp", "get_weather")

        assert result.name == "get_weather"

    async def test_get_connector_tool_not_found(self, connector_service, sample_tool_def):
        """Test getting a non-existent tool raises error."""
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            source=ConnectorSource.config,
        )

        mock_server_info = MagicMock()
        mock_server_info.name = "Server"
        mock_server_info.description = None
        mock_server_info.version = None

        mock_tools_response = MagicMock()
        mock_tools_response.data = [sample_tool_def]

        with patch("llama_stack.core.connectors.connectors._import_mcp_tools") as mock_import:
            mock_get_info = AsyncMock(return_value=mock_server_info)
            mock_list_tools = AsyncMock(return_value=mock_tools_response)
            mock_import.return_value = (mock_get_info, mock_list_tools)

            with pytest.raises(ConnectorToolNotFoundError) as exc_info:
                await connector_service.get_connector_tool("my-mcp", "non_existent_tool")

        assert "my-mcp" in str(exc_info.value)
        assert "non_existent_tool" in str(exc_info.value)


# --- Key prefix tests ---


class TestKeyPrefix:
    """Tests for connector key namespacing."""

    async def test_connectors_use_namespaced_keys(self, connector_service, mock_kvstore):
        """Test that connectors are stored with the correct key prefix."""
        await connector_service.register_connector(
            connector_id="test-connector",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            source=ConnectorSource.config,
        )

        # Check the key was stored with prefix
        keys = list(mock_kvstore._storage.keys())
        assert len(keys) == 1
        assert keys[0] == "connectors:v1:test-connector"


# --- OpenAIResponseInputToolMCP validation tests ---


class TestMCPToolValidation:
    """Tests for MCP tool input validation."""

    def test_mcp_tool_requires_server_url_or_connector_id(self):
        """Test that either server_url or connector_id must be provided."""
        with pytest.raises(ValueError, match="server_url.*connector_id"):
            OpenAIResponseInputToolMCP(
                type="mcp",
                server_label="test",
                # Neither server_url nor connector_id provided
            )

    def test_mcp_tool_accepts_server_url_only(self):
        """Test that server_url alone is valid."""
        tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            server_url="http://localhost:8080/mcp",
        )
        assert tool.server_url == "http://localhost:8080/mcp"
        assert tool.connector_id is None

    def test_mcp_tool_accepts_connector_id_only(self):
        """Test that connector_id alone is valid."""
        tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            connector_id="my-connector",
        )
        assert tool.connector_id == "my-connector"
        assert tool.server_url is None

    def test_mcp_tool_accepts_both_server_url_and_connector_id(self):
        """Test that both can be provided (server_url takes precedence)."""
        tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            server_url="http://localhost:8080/mcp",
            connector_id="my-connector",
        )
        assert tool.server_url == "http://localhost:8080/mcp"
        assert tool.connector_id == "my-connector"


# --- connector_id resolution tests ---


class TestConnectorIdResolution:
    """Tests for the resolve_mcp_connector_id helper function."""

    async def test_connector_id_resolved_to_server_url(self, mock_connectors_api, sample_connector):
        """Test that connector_id is resolved to server_url via connectors API."""
        from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
            resolve_mcp_connector_id,
        )

        mock_connectors_api.get_connector.return_value = sample_connector

        mcp_tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            connector_id="my-mcp-server",
        )

        # Call the actual helper function
        resolved_tool = await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)

        assert resolved_tool.server_url == "http://localhost:8080/mcp"
        mock_connectors_api.get_connector.assert_called_once_with("my-mcp-server")

    async def test_server_url_not_overwritten_when_provided(self, mock_connectors_api):
        """Test that existing server_url is not overwritten even if connector_id provided."""
        from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
            resolve_mcp_connector_id,
        )

        mcp_tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            server_url="http://original-server:8080/mcp",
            connector_id="my-mcp-server",
        )

        # Call the actual helper function
        resolved_tool = await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)

        # Should keep original URL
        assert resolved_tool.server_url == "http://original-server:8080/mcp"
        # Should not call connectors API since server_url already exists
        mock_connectors_api.get_connector.assert_not_called()

    async def test_connector_id_resolution_propagates_not_found_error(self, mock_connectors_api):
        """Test that ConnectorNotFoundError propagates when connector doesn't exist."""
        from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
            resolve_mcp_connector_id,
        )

        mock_connectors_api.get_connector.side_effect = ConnectorNotFoundError("unknown-connector")

        mcp_tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            connector_id="unknown-connector",
        )

        with pytest.raises(ConnectorNotFoundError):
            await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)
