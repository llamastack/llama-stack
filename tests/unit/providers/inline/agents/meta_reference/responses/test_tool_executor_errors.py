# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for tool execution error handling in ToolExecutor.

These tests verify that the error handling code in tool_executor.py correctly handles:
- Tool execution exceptions (MCP tools, web search, knowledge search)
- Vector store failures
- Network timeouts and connection errors
- Invalid tool responses

The implementation has extensive error handling (tool_executor.py:313-378) that was previously
NOT unit tested. These tests ensure that error handling logic is validated.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.providers.inline.agents.meta_reference.responses.tool_executor import (
    ToolExecutor,
)
from llama_stack_api import (
    OpenAIChatCompletionFunction,
    OpenAIChatCompletionToolCall,
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolMCP,
    TextContentItem,
    ToolInvocationResult,
)
from llama_stack_api.openai_responses import (
    ChatCompletionContext,
)


@pytest.fixture
def mock_tool_groups_api():
    """Create mock ToolGroups API."""
    return AsyncMock()


@pytest.fixture
def mock_tool_runtime_api():
    """Create mock ToolRuntime API."""
    return AsyncMock()


@pytest.fixture
def mock_vector_io_api():
    """Create mock VectorIO API."""
    return AsyncMock()


@pytest.fixture
def vector_stores_config():
    """Create VectorStoresConfig with default templates."""
    return VectorStoresConfig()


@pytest.fixture
def tool_executor(mock_tool_groups_api, mock_tool_runtime_api, mock_vector_io_api, vector_stores_config):
    """Create ToolExecutor instance with mocked dependencies."""
    return ToolExecutor(
        tool_groups_api=mock_tool_groups_api,
        tool_runtime_api=mock_tool_runtime_api,
        vector_io_api=mock_vector_io_api,
        vector_stores_config=vector_stores_config,
    )


@pytest.fixture
def chat_completion_context():
    """Create ChatCompletionContext for testing."""
    return ChatCompletionContext(
        response_tools=None,
        tool_choice=None,
        chat_tools=None,
    )


class TestToolExecutionExceptions:
    """Test that tool execution exceptions are captured, not raised."""

    async def test_tool_runtime_exception_captured(self, tool_executor, mock_tool_runtime_api):
        """Test that exceptions from tool_runtime_api.invoke_tool are captured."""
        # Mock tool_runtime_api to raise exception
        mock_tool_runtime_api.invoke_tool.side_effect = RuntimeError("Tool execution failed")

        # Call _execute_tool
        error_exc, result = await tool_executor._execute_tool(
            function_name="test_tool",
            tool_kwargs={"arg1": "value1"},
            ctx=ChatCompletionContext(),
        )

        # Verify exception was captured (not raised)
        assert error_exc is not None
        assert isinstance(error_exc, RuntimeError)
        assert str(error_exc) == "Tool execution failed"
        assert result is None

    async def test_mcp_tool_exception_captured(self, tool_executor):
        """Test that exceptions from MCP tool invocation are captured."""
        # Create MCP tool config
        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"test_mcp_tool": mcp_tool}

        # Mock the invoke_mcp_tool function to raise exception
        with pytest.mock.patch(
            "llama_stack.providers.inline.agents.meta_reference.responses.tool_executor.invoke_mcp_tool",
            side_effect=ConnectionRefusedError("MCP server unavailable"),
        ):
            error_exc, result = await tool_executor._execute_tool(
                function_name="test_mcp_tool",
                tool_kwargs={"param": "value"},
                ctx=ChatCompletionContext(),
                mcp_tool_to_server=mcp_tool_to_server,
            )

        # Verify exception was captured
        assert error_exc is not None
        assert isinstance(error_exc, ConnectionRefusedError)
        assert "MCP server unavailable" in str(error_exc)
        assert result is None

    async def test_knowledge_search_exception_captured(self, tool_executor):
        """Test that exceptions from knowledge_search are captured."""
        # Create file search tool config
        file_search_tool = OpenAIResponseInputToolFileSearch(
            vector_store_ids=["vs_123"],
        )
        ctx = ChatCompletionContext(response_tools=[file_search_tool])

        # Mock vector store search to raise exception
        with pytest.mock.patch.object(
            tool_executor,
            "_execute_knowledge_search_via_vector_store",
            side_effect=TimeoutError("Vector store timeout"),
        ):
            error_exc, result = await tool_executor._execute_tool(
                function_name="knowledge_search",
                tool_kwargs={"query": "test query"},
                ctx=ctx,
            )

        # Verify exception was captured
        assert error_exc is not None
        assert isinstance(error_exc, TimeoutError)
        assert result is None


class TestVectorStoreFailures:
    """Test vector store search failure handling."""

    async def test_vector_store_search_failure_returns_empty(self, tool_executor, mock_vector_io_api):
        """Test that vector store search failures return empty results gracefully."""
        # Create file search tool
        file_search_tool = OpenAIResponseInputToolFileSearch(
            vector_store_ids=["vs_123", "vs_456"],
        )

        # Mock vector store search to fail for first store, succeed for second
        async def mock_search(vector_store_id, request):
            if vector_store_id == "vs_123":
                raise ConnectionError("Vector store connection failed")
            # Return empty result for second store
            return Mock(data=[])

        mock_vector_io_api.openai_search_vector_store = mock_search

        # Execute knowledge search
        result = await tool_executor._execute_knowledge_search_via_vector_store(
            query="test query",
            response_file_search_tool=file_search_tool,
        )

        # Verify we got a result (not an exception)
        assert isinstance(result, ToolInvocationResult)
        assert result.content is not None
        # Should have header, footer, and context templates even with no results
        assert len(result.content) >= 3

    async def test_all_vector_stores_fail_returns_empty(self, tool_executor, mock_vector_io_api):
        """Test that when all vector stores fail, we still return a valid result."""
        # Create file search tool
        file_search_tool = OpenAIResponseInputToolFileSearch(
            vector_store_ids=["vs_123", "vs_456"],
        )

        # Mock all vector store searches to fail
        mock_vector_io_api.openai_search_vector_store.side_effect = TimeoutError("All stores timeout")

        # Execute knowledge search
        result = await tool_executor._execute_knowledge_search_via_vector_store(
            query="test query",
            response_file_search_tool=file_search_tool,
        )

        # Verify we got a result with empty search results
        assert isinstance(result, ToolInvocationResult)
        assert result.content is not None
        # Should have header (with 0 chunks), footer, and context
        assert len(result.content) >= 3
        # Verify metadata shows no results
        assert result.metadata["document_ids"] == []
        assert result.metadata["chunks"] == []
        assert result.metadata["scores"] == []


class TestMCPToolFailures:
    """Test MCP tool execution error handling."""

    async def test_mcp_server_unavailable(self, tool_executor):
        """Test MCP tool execution when server is unavailable."""
        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"get_data": mcp_tool}

        # Mock invoke_mcp_tool to raise ConnectionRefusedError
        with pytest.mock.patch(
            "llama_stack.providers.inline.agents.meta_reference.responses.tool_executor.invoke_mcp_tool",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            error_exc, result = await tool_executor._execute_tool(
                function_name="get_data",
                tool_kwargs={},
                ctx=ChatCompletionContext(),
                mcp_tool_to_server=mcp_tool_to_server,
            )

        assert error_exc is not None
        assert isinstance(error_exc, ConnectionRefusedError)
        assert result is None

    async def test_mcp_tool_timeout(self, tool_executor):
        """Test MCP tool execution timeout."""
        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"slow_tool": mcp_tool}

        # Mock invoke_mcp_tool to raise TimeoutError
        with pytest.mock.patch(
            "llama_stack.providers.inline.agents.meta_reference.responses.tool_executor.invoke_mcp_tool",
            side_effect=TimeoutError("MCP call timeout"),
        ):
            error_exc, result = await tool_executor._execute_tool(
                function_name="slow_tool",
                tool_kwargs={},
                ctx=ChatCompletionContext(),
                mcp_tool_to_server=mcp_tool_to_server,
            )

        assert error_exc is not None
        assert isinstance(error_exc, TimeoutError)
        assert result is None

    async def test_mcp_tool_returns_error_result(self, tool_executor):
        """Test MCP tool that returns a result with error_code/error_message."""
        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"error_tool": mcp_tool}

        # Mock invoke_mcp_tool to return error result
        error_result = Mock()
        error_result.error_code = 500
        error_result.error_message = "Internal tool error"
        error_result.content = None

        with pytest.mock.patch(
            "llama_stack.providers.inline.agents.meta_reference.responses.tool_executor.invoke_mcp_tool",
            return_value=error_result,
        ):
            error_exc, result = await tool_executor._execute_tool(
                function_name="error_tool",
                tool_kwargs={},
                ctx=ChatCompletionContext(),
                mcp_tool_to_server=mcp_tool_to_server,
            )

        # No exception raised, but result contains error
        assert error_exc is None
        assert result is not None
        assert result.error_code == 500
        assert result.error_message == "Internal tool error"


class TestWebSearchFailures:
    """Test web search tool error handling."""

    async def test_web_search_api_failure(self, tool_executor, mock_tool_runtime_api):
        """Test web search tool when API fails."""
        # Mock tool_runtime_api to raise exception for web_search
        mock_tool_runtime_api.invoke_tool.side_effect = ConnectionError("Web search API unavailable")

        error_exc, result = await tool_executor._execute_tool(
            function_name="web_search",
            tool_kwargs={"query": "test query"},
            ctx=ChatCompletionContext(),
        )

        # Exception should be captured
        assert error_exc is not None
        assert isinstance(error_exc, ConnectionError)
        assert "Web search API unavailable" in str(error_exc)
        assert result is None

    async def test_web_search_returns_invalid_data(self, tool_executor, mock_tool_runtime_api):
        """Test web search tool when it returns invalid/malformed data."""
        # Mock tool_runtime_api to return invalid result
        invalid_result = Mock()
        invalid_result.content = "Not a valid format"  # Should be list, not string
        invalid_result.error_code = None
        invalid_result.error_message = None

        mock_tool_runtime_api.invoke_tool.return_value = invalid_result

        error_exc, result = await tool_executor._execute_tool(
            function_name="web_search",
            tool_kwargs={"query": "test"},
            ctx=ChatCompletionContext(),
        )

        # Should succeed (error handling happens at message building stage)
        assert error_exc is None
        assert result is not None


class TestErrorMessageBuilding:
    """Test error message construction from tool execution failures."""

    async def test_error_message_from_exception(self, tool_executor):
        """Test that exception messages are properly included in output."""
        # Create tool call
        tool_call = OpenAIChatCompletionToolCall(
            id="call_123",
            function=OpenAIChatCompletionFunction(
                name="failing_tool",
                arguments=json.dumps({"arg": "value"}),
            ),
        )
        ctx = ChatCompletionContext()

        # Mock tool execution to raise exception
        with pytest.mock.patch.object(tool_executor, "_execute_tool", return_value=(RuntimeError("Tool failed"), None)):
            # Execute tool call
            results = []
            async for result in tool_executor.execute_tool_call(
                tool_call=tool_call,
                ctx=ctx,
                sequence_number=1,
                output_index=0,
                item_id="item_123",
            ):
                results.append(result)

            # Get final result
            final_result = results[-1]
            assert final_result.final_input_message is not None
            assert "Tool failed" in final_result.final_input_message.content

    async def test_error_message_from_mcp_error_result(self, tool_executor):
        """Test that MCP error results are properly formatted in output."""
        # Create MCP tool call
        tool_call = OpenAIChatCompletionToolCall(
            id="call_123",
            function=OpenAIChatCompletionFunction(
                name="mcp_error_tool",
                arguments=json.dumps({}),
            ),
        )

        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"mcp_error_tool": mcp_tool}

        ctx = ChatCompletionContext()

        # Mock tool execution to return error result
        error_result = Mock()
        error_result.error_code = 404
        error_result.error_message = "Resource not found"
        error_result.content = None

        with pytest.mock.patch.object(tool_executor, "_execute_tool", return_value=(None, error_result)):
            # Execute tool call
            results = []
            async for result in tool_executor.execute_tool_call(
                tool_call=tool_call,
                ctx=ctx,
                sequence_number=1,
                output_index=0,
                item_id="item_123",
                mcp_tool_to_server=mcp_tool_to_server,
            ):
                results.append(result)

            # Get final result
            final_result = results[-1]
            assert final_result.final_output_message is not None
            # Error should be in the output message
            assert final_result.final_output_message.error is not None
            assert "404" in final_result.final_output_message.error
            assert "Resource not found" in final_result.final_output_message.error


class TestNetworkFailures:
    """Test network error handling."""

    async def test_connection_timeout(self, tool_executor, mock_tool_runtime_api):
        """Test handling of network timeout errors."""
        mock_tool_runtime_api.invoke_tool.side_effect = TimeoutError("Connection timeout")

        error_exc, result = await tool_executor._execute_tool(
            function_name="network_tool",
            tool_kwargs={},
            ctx=ChatCompletionContext(),
        )

        assert error_exc is not None
        assert isinstance(error_exc, TimeoutError)
        assert result is None

    async def test_dns_resolution_failure(self, tool_executor, mock_tool_runtime_api):
        """Test handling of DNS resolution failures."""
        mock_tool_runtime_api.invoke_tool.side_effect = OSError("Name or service not known")

        error_exc, result = await tool_executor._execute_tool(
            function_name="remote_tool",
            tool_kwargs={},
            ctx=ChatCompletionContext(),
        )

        assert error_exc is not None
        assert isinstance(error_exc, OSError)
        assert result is None


class TestToolExecutionEventEmission:
    """Test that proper events are emitted during tool execution failures."""

    async def test_mcp_failed_event_emitted(self, tool_executor):
        """Test that MCP failed event is emitted when tool fails."""
        # Create MCP tool
        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"failing_mcp_tool": mcp_tool}

        # Mock tool execution to raise exception
        with pytest.mock.patch.object(
            tool_executor, "_execute_tool", return_value=(RuntimeError("MCP tool failed"), None)
        ):
            # Execute tool call
            tool_call = OpenAIChatCompletionToolCall(
                id="call_123",
                function=OpenAIChatCompletionFunction(
                    name="failing_mcp_tool",
                    arguments=json.dumps({}),
                ),
            )

            results = []
            async for result in tool_executor.execute_tool_call(
                tool_call=tool_call,
                ctx=ChatCompletionContext(),
                sequence_number=1,
                output_index=0,
                item_id="item_123",
                mcp_tool_to_server=mcp_tool_to_server,
            ):
                results.append(result)

            # Check for failed event
            failed_events = [r for r in results if r.stream_event and r.stream_event.type == "response.mcp_call.failed"]
            assert len(failed_events) > 0, "Should emit mcp_call.failed event"

    async def test_no_failed_event_for_successful_execution(self, tool_executor):
        """Test that no failed event is emitted when tool succeeds."""
        # Create MCP tool
        mcp_tool = OpenAIResponseInputToolMCP(
            server_label="test_server",
            server_url="http://localhost:8080",
        )
        mcp_tool_to_server = {"success_tool": mcp_tool}

        # Mock successful tool execution
        success_result = Mock()
        success_result.error_code = None
        success_result.error_message = None
        success_result.content = [TextContentItem(text="Success result")]

        with pytest.mock.patch.object(tool_executor, "_execute_tool", return_value=(None, success_result)):
            # Execute tool call
            tool_call = OpenAIChatCompletionToolCall(
                id="call_123",
                function=OpenAIChatCompletionFunction(
                    name="success_tool",
                    arguments=json.dumps({}),
                ),
            )

            results = []
            async for result in tool_executor.execute_tool_call(
                tool_call=tool_call,
                ctx=ChatCompletionContext(),
                sequence_number=1,
                output_index=0,
                item_id="item_123",
                mcp_tool_to_server=mcp_tool_to_server,
            ):
                results.append(result)

            # Check for completed event (not failed)
            completed_events = [
                r for r in results if r.stream_event and r.stream_event.type == "response.mcp_call.completed"
            ]
            failed_events = [r for r in results if r.stream_event and r.stream_event.type == "response.mcp_call.failed"]

            assert len(completed_events) > 0, "Should emit mcp_call.completed event"
            assert len(failed_events) == 0, "Should NOT emit mcp_call.failed event"
