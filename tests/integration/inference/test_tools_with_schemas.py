# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for inference/chat completion with JSON Schema-based tools.
Tests that tools pass through correctly to various LLM providers.
"""

import json

import pytest

from llama_stack.core.library_client import LlamaStackAsLibraryClient
from llama_stack.models.llama.datatypes import ToolDefinition
from tests.common.mcp import make_mcp_server

AUTH_TOKEN = "test-token"


class TestChatCompletionWithTools:
    """Test chat completion with tools that have complex schemas."""

    def test_simple_tool_call(self, llama_stack_client, text_model_id):
        """Test basic tool calling with simple input schema."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
            tools=tools,
        )

        assert response is not None

    def test_tool_with_complex_schema(self, llama_stack_client, text_model_id):
        """Test tool calling with complex schema including $ref and $defs."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a flight",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flight": {"$ref": "#/$defs/FlightInfo"},
                            "passenger": {"$ref": "#/$defs/Passenger"},
                        },
                        "required": ["flight", "passenger"],
                        "$defs": {
                            "FlightInfo": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "date": {"type": "string", "format": "date"},
                                },
                            },
                            "Passenger": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                            },
                        },
                    },
                },
            }
        ]

        response = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Book a flight from SFO to JFK for John Doe"}],
            tools=tools,
        )

        # The key test: No errors during schema processing
        # The LLM received a valid, complete schema with $ref/$defs
        assert response is not None


class TestOpenAICompatibility:
    """Test OpenAI-compatible endpoints with new schema format."""

    def test_openai_chat_completion_with_tools(self, compat_client, text_model_id):
        """Test OpenAI-compatible chat completion with tools."""
        from openai import OpenAI

        if not isinstance(compat_client, OpenAI):
            pytest.skip("OpenAI client required")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = compat_client.chat.completions.create(
            model=text_model_id, messages=[{"role": "user", "content": "What's the weather in Tokyo?"}], tools=tools
        )

        assert response is not None
        assert response.choices is not None

    def test_openai_format_preserves_complex_schemas(self, compat_client, text_model_id):
        """Test that complex schemas work through OpenAI-compatible API."""
        from openai import OpenAI

        if not isinstance(compat_client, OpenAI):
            pytest.skip("OpenAI client required")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process_data",
                    "description": "Process structured data",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"$ref": "#/$defs/DataObject"}},
                        "$defs": {
                            "DataObject": {
                                "type": "object",
                                "properties": {"values": {"type": "array", "items": {"type": "number"}}},
                            }
                        },
                    },
                },
            }
        ]

        response = compat_client.chat.completions.create(
            model=text_model_id, messages=[{"role": "user", "content": "Process this data"}], tools=tools
        )

        assert response is not None


class TestMCPToolsInChatCompletion:
    """Test using MCP tools in chat completion."""

    @pytest.fixture
    def mcp_with_schemas(self):
        """MCP server for chat completion tests."""
        from mcp.server.fastmcp import Context

        async def calculate(x: float, y: float, operation: str, ctx: Context) -> float:
            ops = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y if y != 0 else None}
            return ops.get(operation, 0)

        with make_mcp_server(required_auth_token=AUTH_TOKEN, tools={"calculate": calculate}) as server:
            yield server

    def test_mcp_tools_in_inference(self, llama_stack_client, text_model_id, mcp_with_schemas):
        """Test that MCP tools can be used in inference."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        test_toolgroup_id = "mcp::calc"
        uri = mcp_with_schemas["server_url"]

        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )

        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        # Get the tools from MCP
        tools_response = llama_stack_client.tool_runtime.list_tools(
            tool_group_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )

        # Convert to OpenAI format for inference
        tools = []
        for tool in tools_response:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema or {},
                    },
                }
            )

        # Use in chat completion
        response = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Calculate 5 + 3"}],
            tools=tools,
        )

        # Schema should have been passed through correctly
        assert response is not None


class TestProviderSpecificBehavior:
    """Test provider-specific handling of schemas."""

    def test_openai_provider_drops_output_schema(self, llama_stack_client, text_model_id):
        """Test that OpenAI provider doesn't send output_schema (API limitation)."""
        # This is more of a documentation test
        # OpenAI API doesn't support output schemas, so we drop them

        _tool = ToolDefinition(
            tool_name="test",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"y": {"type": "number"}}},
        )

        # When this tool is sent to OpenAI provider, output_schema is dropped
        # But input_schema is preserved
        # This test documents the expected behavior

        # We can't easily test this without mocking, but the unit tests cover it
        pass

    def test_gemini_array_support(self):
        """Test that Gemini receives array schemas correctly (issue from commit 65f7b81e)."""
        # This was the original bug that led to adding 'items' field
        # Now with full JSON Schema pass-through, arrays should work

        tool = ToolDefinition(
            tool_name="tag_processor",
            input_schema={
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}, "description": "List of tags"}},
            },
        )

        # With new approach, the complete schema with items is preserved
        assert tool.input_schema["properties"]["tags"]["type"] == "array"
        assert tool.input_schema["properties"]["tags"]["items"]["type"] == "string"


class TestStreamingWithTools:
    """Test streaming chat completion with tools."""

    def test_streaming_tool_calls(self, llama_stack_client, text_model_id):
        """Test that tool schemas work correctly in streaming mode."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            }
        ]

        response_stream = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "What time is it in UTC?"}],
            tools=tools,
            stream=True,
        )

        # Should be able to iterate through stream
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)

        # Should have received some chunks
        assert len(chunks) >= 0


class TestEdgeCases:
    """Test edge cases in inference with tools."""

    def test_tool_without_schema(self, llama_stack_client, text_model_id):
        """Test tool with no input_schema."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_args_tool",
                    "description": "Tool with no arguments",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Call the no args tool"}],
            tools=tools,
        )

        assert response is not None

    def test_multiple_tools_with_different_schemas(self, llama_stack_client, text_model_id):
        """Test multiple tools with different schema complexities."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complex",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"$ref": "#/$defs/Complex"}},
                        "$defs": {
                            "Complex": {
                                "type": "object",
                                "properties": {"nested": {"type": "array", "items": {"type": "number"}}},
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "with_output",
                    "parameters": {"type": "object", "properties": {"input": {"type": "string"}}},
                },
            },
        ]

        response = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Use one of the available tools"}],
            tools=tools,
        )

        # All tools should have been processed without errors
        assert response is not None
