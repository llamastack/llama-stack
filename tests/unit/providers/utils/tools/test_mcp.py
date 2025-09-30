# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, Mock, patch

from llama_stack.apis.tools import ListToolDefsResponse
from llama_stack.providers.utils.tools.mcp import (
    MCPProtol,
    list_mcp_tools,
    resolve_json_schema_refs,
)


class TestResolveJsonSchemaRefs:
    """Test cases for resolve_json_schema_refs function."""

    def test_resolve_simple_ref(self):
        """Test resolving a simple $ref reference."""
        schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "$defs": {
                "User": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
            },
        }

        result = resolve_json_schema_refs(schema)

        expected = {
            "type": "object",
            "properties": {
                "user": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
            },
        }

        assert result == expected

    def test_resolve_nested_refs(self):
        """Test resolving nested $ref references."""
        schema = {
            "type": "object",
            "properties": {"data": {"$ref": "#/$defs/Container"}},
            "$defs": {
                "Container": {"type": "object", "properties": {"user": {"$ref": "#/$defs/User"}}},
                "User": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }

        result = resolve_json_schema_refs(schema)

        expected = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {"user": {"type": "object", "properties": {"name": {"type": "string"}}}},
                }
            },
        }

        assert result == expected

    def test_resolve_refs_in_array(self):
        """Test resolving $ref references within arrays."""
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
            "$defs": {"Item": {"type": "object", "properties": {"id": {"type": "string"}}}},
        }

        result = resolve_json_schema_refs(schema)

        expected = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}}
            },
        }

        assert result == expected

    def test_resolve_missing_ref(self):
        """Test handling of missing $ref definition."""
        schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/MissingUser"}},
            "$defs": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }

        with patch("llama_stack.providers.utils.tools.mcp.logger") as mock_logger:
            result = resolve_json_schema_refs(schema)
            mock_logger.warning.assert_called_once_with("Referenced definition 'MissingUser' not found in $defs")

        # Should return the original $ref unchanged
        expected = {"type": "object", "properties": {"user": {"$ref": "#/$defs/MissingUser"}}}

        assert result == expected

    def test_resolve_unsupported_ref_format(self):
        """Test handling of unsupported $ref format."""
        schema = {"type": "object", "properties": {"user": {"$ref": "http://example.com/schema"}}, "$defs": {}}

        with patch("llama_stack.providers.utils.tools.mcp.logger") as mock_logger:
            result = resolve_json_schema_refs(schema)
            mock_logger.warning.assert_called_once_with("Unsupported $ref format: http://example.com/schema")

        # Should return the original $ref unchanged
        expected = {"type": "object", "properties": {"user": {"$ref": "http://example.com/schema"}}}

        assert result == expected

    def test_resolve_no_defs(self):
        """Test schema without $defs section."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = resolve_json_schema_refs(schema)

        assert result == schema

    def test_resolve_non_dict_input(self):
        """Test with non-dictionary input."""
        assert resolve_json_schema_refs("string") == "string"
        assert resolve_json_schema_refs(123) == 123
        assert resolve_json_schema_refs(["list"]) == ["list"]
        assert resolve_json_schema_refs(None) is None

    def test_resolve_preserves_original(self):
        """Test that original schema is not modified."""
        original_schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "$defs": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }

        resolve_json_schema_refs(original_schema)

        # Original should be unchanged (but this is a shallow comparison)
        assert "$ref" in original_schema["properties"]["user"]
        assert "$defs" in original_schema


class TestMCPProtocol:
    """Test cases for MCPProtol enum."""

    def test_protocol_values(self):
        """Test enum values are correct."""
        assert MCPProtol.UNKNOWN.value == 0
        assert MCPProtol.STREAMABLE_HTTP.value == 1
        assert MCPProtol.SSE.value == 2


class TestListMcpTools:
    """Test cases for list_mcp_tools function."""

    async def test_list_tools_success(self):
        """Test successful listing of MCP tools."""
        endpoint = "http://example.com/mcp"
        headers = {"Authorization": "Bearer token"}

        # Mock tool from MCP
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter", "default": "default_value"},
                "param2": {"type": "integer", "description": "Second parameter"},
            },
        }

        mock_tools_result = Mock()
        mock_tools_result.tools = [mock_tool]

        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        with patch("llama_stack.providers.utils.tools.mcp.client_wrapper") as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()

            result = await list_mcp_tools(endpoint, headers)

            assert isinstance(result, ListToolDefsResponse)
            assert len(result.data) == 1

            tool_def = result.data[0]
            assert tool_def.name == "test_tool"
            assert tool_def.description == "A test tool"
            assert tool_def.metadata["endpoint"] == endpoint

            # Check parameters
            assert len(tool_def.parameters) == 2

            param1 = next(p for p in tool_def.parameters if p.name == "param1")
            assert param1.parameter_type == "string"
            assert param1.description == "First parameter"
            assert param1.required is False  # Has default value
            assert param1.default == "default_value"

            param2 = next(p for p in tool_def.parameters if p.name == "param2")
            assert param2.parameter_type == "integer"
            assert param2.description == "Second parameter"
            assert param2.required is True  # No default value

    async def test_list_tools_with_schema_refs(self):
        """Test listing tools with JSON Schema $refs."""
        endpoint = "http://example.com/mcp"
        headers = {}

        # Mock tool with $ref in schema
        mock_tool = Mock()
        mock_tool.name = "ref_tool"
        mock_tool.description = "Tool with refs"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "$defs": {
                "User": {"type": "object", "properties": {"name": {"type": "string", "description": "User name"}}}
            },
        }

        mock_tools_result = Mock()
        mock_tools_result.tools = [mock_tool]

        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        with patch("llama_stack.providers.utils.tools.mcp.client_wrapper") as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()

            result = await list_mcp_tools(endpoint, headers)

            # Should have resolved the $ref
            tool_def = result.data[0]
            assert len(tool_def.parameters) == 1

            # The user parameter should be flattened from the resolved $ref
            # Note: This depends on how the schema resolution works with nested objects

    async def test_list_tools_empty_result(self):
        """Test listing tools when no tools are available."""
        endpoint = "http://example.com/mcp"
        headers = {}

        mock_tools_result = Mock()
        mock_tools_result.tools = []

        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        with patch("llama_stack.providers.utils.tools.mcp.client_wrapper") as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()

            result = await list_mcp_tools(endpoint, headers)

            assert isinstance(result, ListToolDefsResponse)
            assert len(result.data) == 0
