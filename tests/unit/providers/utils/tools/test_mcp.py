# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
import httpx
from mcp import McpError
from mcp import types as mcp_types

from llama_stack.providers.utils.tools.mcp import (
    resolve_json_schema_refs,
    MCPProtol,
    client_wrapper,
    list_mcp_tools,
    invoke_mcp_tool,
    protocol_cache,
)
from llama_stack.apis.tools import ListToolDefsResponse, ToolDef, ToolParameter, ToolInvocationResult
from llama_stack.apis.common.content_types import TextContentItem, ImageContentItem
from llama_stack.core.datatypes import AuthenticationRequiredError


class TestResolveJsonSchemaRefs:
    """Test cases for resolve_json_schema_refs function."""

    def test_resolve_simple_ref(self):
        """Test resolving a simple $ref reference."""
        schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/$defs/User"}
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        result = resolve_json_schema_refs(schema)
        
        expected = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        assert result == expected

    def test_resolve_nested_refs(self):
        """Test resolving nested $ref references."""
        schema = {
            "type": "object",
            "properties": {
                "data": {"$ref": "#/$defs/Container"}
            },
            "$defs": {
                "Container": {
                    "type": "object",
                    "properties": {
                        "user": {"$ref": "#/$defs/User"}
                    }
                },
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        result = resolve_json_schema_refs(schema)
        
        expected = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        
        assert result == expected

    def test_resolve_refs_in_array(self):
        """Test resolving $ref references within arrays."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"}
                }
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    }
                }
            }
        }
        
        result = resolve_json_schema_refs(schema)
        
        expected = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        assert result == expected

    def test_resolve_missing_ref(self):
        """Test handling of missing $ref definition."""
        schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/$defs/MissingUser"}
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        with patch('llama_stack.providers.utils.tools.mcp.logger') as mock_logger:
            result = resolve_json_schema_refs(schema)
            mock_logger.warning.assert_called_once_with("Referenced definition 'MissingUser' not found in $defs")
        
        # Should return the original $ref unchanged
        expected = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/$defs/MissingUser"}
            }
        }
        
        assert result == expected

    def test_resolve_unsupported_ref_format(self):
        """Test handling of unsupported $ref format."""
        schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "http://example.com/schema"}
            },
            "$defs": {}
        }
        
        with patch('llama_stack.providers.utils.tools.mcp.logger') as mock_logger:
            result = resolve_json_schema_refs(schema)
            mock_logger.warning.assert_called_once_with("Unsupported $ref format: http://example.com/schema")
        
        # Should return the original $ref unchanged
        expected = {
            "type": "object",
            "properties": {
                "user": {"$ref": "http://example.com/schema"}
            }
        }
        
        assert result == expected

    def test_resolve_no_defs(self):
        """Test schema without $defs section."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
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
            "properties": {
                "user": {"$ref": "#/$defs/User"}
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        original_copy = original_schema.copy()
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


class TestClientWrapper:
    """Test cases for client_wrapper function."""

    @pytest.fixture
    def mock_client_session(self):
        """Mock ClientSession for testing."""
        session = Mock()
        session.initialize = AsyncMock()
        return session

    @pytest.fixture
    def mock_client_streams(self):
        """Mock client streams."""
        return (Mock(), Mock())

    @pytest.mark.asyncio
    async def test_successful_streamable_http_connection(self, mock_client_session, mock_client_streams):
        """Test successful connection with STREAMABLE_HTTP protocol."""
        endpoint = "http://example.com/mcp"
        headers = {"Authorization": "Bearer token"}
        
        # Create a proper context manager mock
        mock_http_context = AsyncMock()
        mock_http_context.__aenter__ = AsyncMock(return_value=mock_client_streams)
        mock_http_context.__aexit__ = AsyncMock(return_value=False)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_client_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=False)
        
        with patch('llama_stack.providers.utils.tools.mcp.streamablehttp_client') as mock_http_client, \
             patch('llama_stack.providers.utils.tools.mcp.ClientSession') as mock_session_class:
            
            mock_http_client.return_value = mock_http_context
            mock_session_class.return_value = mock_session_context
            
            async with client_wrapper(endpoint, headers) as session:
                assert session == mock_client_session
                mock_client_session.initialize.assert_called_once()
                assert protocol_cache.get(endpoint) == MCPProtol.STREAMABLE_HTTP

 
    @pytest.mark.asyncio
    async def test_cached_protocol_preference(self, mock_client_session, mock_client_streams):
        """Test that cached protocol is tried first."""
        endpoint = "http://example.com/mcp"
        headers = {"Authorization": "Bearer token"}
        
        # Set SSE as cached protocol
        protocol_cache[endpoint] = MCPProtol.SSE
        
        # Create proper context manager mocks
        mock_sse_context = AsyncMock()
        mock_sse_context.__aenter__ = AsyncMock(return_value=mock_client_streams)
        mock_sse_context.__aexit__ = AsyncMock(return_value=False)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_client_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=False)
        
        with patch('llama_stack.providers.utils.tools.mcp.sse_client') as mock_sse_client, \
             patch('llama_stack.providers.utils.tools.mcp.streamablehttp_client') as mock_http_client, \
             patch('llama_stack.providers.utils.tools.mcp.ClientSession') as mock_session_class:
            
            mock_sse_client.return_value = mock_sse_context
            mock_session_class.return_value = mock_session_context
            
            async with client_wrapper(endpoint, headers) as session:
                assert session == mock_client_session
                # SSE should be tried first due to cache
                mock_sse_client.assert_called_once()
                mock_http_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_authentication_error_raises_exception(self):
        """Test that 401 errors raise AuthenticationRequiredError."""
        endpoint = "http://example.com/mcp"
        headers = {"Authorization": "Bearer invalid"}
        
        protocol_cache.clear()
        
        # Create a proper HTTP 401 error
        response = Mock()
        response.status_code = 401
        http_error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=response)
        
        with patch('llama_stack.providers.utils.tools.mcp.streamablehttp_client') as mock_http_client:
            mock_http_client.return_value.__aenter__ = AsyncMock(side_effect=http_error)
            mock_http_client.return_value.__aexit__ = AsyncMock()
            
            with pytest.raises(AuthenticationRequiredError):
                async with client_wrapper(endpoint, headers):
                    pass

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        endpoint = "http://example.com/mcp"
        headers = {}
        
        protocol_cache.clear()
        
        connect_error = httpx.ConnectError("Connection refused")
        
        with patch('llama_stack.providers.utils.tools.mcp.streamablehttp_client') as mock_http_client, \
             patch('llama_stack.providers.utils.tools.mcp.sse_client') as mock_sse_client:
            
            mock_http_client.return_value.__aenter__ = AsyncMock(side_effect=connect_error)
            mock_http_client.return_value.__aexit__ = AsyncMock()
            mock_sse_client.return_value.__aenter__ = AsyncMock(side_effect=connect_error)
            mock_sse_client.return_value.__aexit__ = AsyncMock()
            
            with pytest.raises(ConnectionError, match="Failed to connect to MCP server"):
                async with client_wrapper(endpoint, headers):
                    pass

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        endpoint = "http://example.com/mcp"
        headers = {}
        
        protocol_cache.clear()
        
        timeout_error = httpx.TimeoutException("Request timeout")
        
        with patch('llama_stack.providers.utils.tools.mcp.streamablehttp_client') as mock_http_client, \
             patch('llama_stack.providers.utils.tools.mcp.sse_client') as mock_sse_client:
            
            mock_http_client.return_value.__aenter__ = AsyncMock(side_effect=timeout_error)
            mock_http_client.return_value.__aexit__ = AsyncMock()
            mock_sse_client.return_value.__aenter__ = AsyncMock(side_effect=timeout_error)
            mock_sse_client.return_value.__aexit__ = AsyncMock()
            
            with pytest.raises(TimeoutError, match="MCP server.*timed out"):
                async with client_wrapper(endpoint, headers):
                    pass


class TestListMcpTools:
    """Test cases for list_mcp_tools function."""

    @pytest.mark.asyncio
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
                "param1": {
                    "type": "string",
                    "description": "First parameter",
                    "default": "default_value"
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter"
                }
            }
        }
        
        mock_tools_result = Mock()
        mock_tools_result.tools = [mock_tool]
        
        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        
        with patch('llama_stack.providers.utils.tools.mcp.client_wrapper') as mock_wrapper:
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

    @pytest.mark.asyncio
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
            "properties": {
                "user": {"$ref": "#/$defs/User"}
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "User name"}
                    }
                }
            }
        }
        
        mock_tools_result = Mock()
        mock_tools_result.tools = [mock_tool]
        
        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        
        with patch('llama_stack.providers.utils.tools.mcp.client_wrapper') as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()
            
            result = await list_mcp_tools(endpoint, headers)
            
            # Should have resolved the $ref
            tool_def = result.data[0]
            assert len(tool_def.parameters) == 1
            
            # The user parameter should be flattened from the resolved $ref
            # Note: This depends on how the schema resolution works with nested objects

    @pytest.mark.asyncio
    async def test_list_tools_empty_result(self):
        """Test listing tools when no tools are available."""
        endpoint = "http://example.com/mcp"
        headers = {}
        
        mock_tools_result = Mock()
        mock_tools_result.tools = []
        
        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        
        with patch('llama_stack.providers.utils.tools.mcp.client_wrapper') as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()
            
            result = await list_mcp_tools(endpoint, headers)
            
            assert isinstance(result, ListToolDefsResponse)
            assert len(result.data) == 0


class TestInvokeMcpTool:
    """Test cases for invoke_mcp_tool function."""

    @pytest.mark.asyncio
    async def test_invoke_tool_success_with_text_content(self):
        """Test successful tool invocation with text content."""
        endpoint = "http://example.com/mcp"
        headers = {}
        tool_name = "test_tool"
        kwargs = {"param1": "value1", "param2": 42}
        
        # Mock MCP text content
        mock_text_content = mcp_types.TextContent(type="text", text="Tool output text")
        
        mock_result = Mock()
        mock_result.content = [mock_text_content]
        mock_result.isError = False
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('llama_stack.providers.utils.tools.mcp.client_wrapper') as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()
            
            result = await invoke_mcp_tool(endpoint, headers, tool_name, kwargs)
            
            assert isinstance(result, ToolInvocationResult)
            assert result.error_code == 0
            assert len(result.content) == 1
            
            content_item = result.content[0]
            assert isinstance(content_item, TextContentItem)
            assert content_item.text == "Tool output text"
            
            mock_session.call_tool.assert_called_once_with(tool_name, kwargs)

    @pytest.mark.asyncio
    async def test_invoke_tool_with_error(self):
        """Test tool invocation when tool returns an error."""
        endpoint = "http://example.com/mcp"
        headers = {}
        tool_name = "error_tool"
        kwargs = {}
        
        mock_text_content = mcp_types.TextContent(type="text", text="Error occurred")
        
        mock_result = Mock()
        mock_result.content = [mock_text_content]
        mock_result.isError = True
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('llama_stack.providers.utils.tools.mcp.client_wrapper') as mock_wrapper:
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()
            
            result = await invoke_mcp_tool(endpoint, headers, tool_name, kwargs)
            
            assert isinstance(result, ToolInvocationResult)
            assert result.error_code == 1
            assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_invoke_tool_with_embedded_resource_warning(self):
        """Test tool invocation with unsupported EmbeddedResource content."""
        endpoint = "http://example.com/mcp"
        headers = {}
        tool_name = "resource_tool"
        kwargs = {}
        
        # Mock MCP embedded resource content
        mock_embedded_resource = mcp_types.EmbeddedResource(
            type="resource",
            resource={
                "uri": "file:///example.txt",
                "text": "Resource content"
            }
        )
        
        mock_result = Mock()
        mock_result.content = [mock_embedded_resource]
        mock_result.isError = False
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        with patch('llama_stack.providers.utils.tools.mcp.client_wrapper') as mock_wrapper, \
             patch('llama_stack.providers.utils.tools.mcp.logger') as mock_logger:
            
            mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_wrapper.return_value.__aexit__ = AsyncMock()
            
            result = await invoke_mcp_tool(endpoint, headers, tool_name, kwargs)
            
            assert isinstance(result, ToolInvocationResult)
            assert result.error_code == 0
            assert len(result.content) == 0  # EmbeddedResource is skipped
            
            # Should log a warning
            mock_logger.warning.assert_called_once()
            assert "EmbeddedResource is not supported" in str(mock_logger.warning.call_args)

 

@pytest.fixture(autouse=True)
def clear_protocol_cache():
    """Clear protocol cache before each test."""
    protocol_cache.clear()
    yield
    protocol_cache.clear()
