# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack import LlamaStackAsLibraryClient
from tests.common.mcp import make_mcp_server

from .helpers import setup_mcp_tools


# Skip these tests in replay mode until recordings are generated
pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMA_STACK_TEST_INFERENCE_MODE") == "replay",
    reason="No recordings yet for JWT passthrough tests. Run with --inference-mode=record-if-missing to generate.",
)


def test_mcp_jwt_passthrough_basic(compat_client, text_model_id, caplog):
    """
    Test that JWT token is forwarded to MCP server for both list_tools and invoke_tool.

    This is the core test verifying JWT passthrough functionality works end-to-end.
    """
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    # Create MCP server that requires JWT authentication
    test_token = "test-jwt-token-123"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "localmcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "headers": {"Authorization": f"Bearer {test_token}"},
                }
            ],
            mcp_server_info,
        )

        # Create response - JWT should be forwarded
        response = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify list_tools succeeded
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert response.output[0].server_label == "localmcp"
        assert len(response.output[0].tools) == 2

        # Verify tool invocation succeeded
        assert response.output[1].type == "mcp_call"
        assert response.output[1].name == "get_boiling_point"
        assert response.output[1].error is None
        assert "-100" in response.output[1].output


def test_mcp_jwt_passthrough_backward_compatibility(compat_client, text_model_id):
    """
    Test that MCP tools work without JWT (backward compatibility).

    This ensures systems without JWT continue working as before.
    """
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    # Create MCP server WITHOUT authentication requirement
    with make_mcp_server(required_auth_token=None) as mcp_server_info:
        tools = setup_mcp_tools(
            [{"type": "mcp", "server_label": "localmcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}], mcp_server_info
        )

        # Create response without JWT - should still work
        response = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify operation succeeded without JWT
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None


def test_mcp_jwt_passthrough_streaming(compat_client, text_model_id, caplog):
    """
    Test that JWT token is forwarded during streaming responses.
    """
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    # Create MCP server that requires JWT authentication
    test_token = "test-streaming-jwt"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "localmcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "headers": {"Authorization": f"Bearer {test_token}"},
                }
            ],
            mcp_server_info,
        )

        # Create streaming response - JWT should be forwarded
        stream = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=True,
        )

        # Collect all streaming chunks
        chunks = list(stream)
        assert len(chunks) > 0

        # Get final response
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, "response")
        final_response = final_chunk.response

        # Verify MCP operations succeeded
        assert len(final_response.output) >= 3
        assert final_response.output[0].type == "mcp_list_tools"
        assert final_response.output[1].type == "mcp_call"
        assert final_response.output[1].error is None
