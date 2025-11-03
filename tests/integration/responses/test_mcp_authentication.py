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
    reason="No recordings yet for authentication tests. Run with --inference-mode=record-if-missing to generate.",
)


def test_mcp_authentication_bearer(compat_client, text_model_id):
    """Test that bearer authentication is correctly applied to MCP requests."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    test_token = "test-bearer-token-789"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "auth-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "authentication": {
                        "type": "bearer",
                        "token": test_token,
                    },
                }
            ],
            mcp_server_info,
        )

        # Create response - authentication should be applied
        response = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify list_tools succeeded (requires auth)
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert len(response.output[0].tools) == 2

        # Verify tool invocation succeeded (requires auth)
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None


def test_mcp_authentication_different_token(compat_client, text_model_id):
    """Test authentication with a different bearer token."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    test_token = "different-token-456"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "auth2-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "authentication": {
                        "type": "bearer",
                        "token": test_token,
                    },
                }
            ],
            mcp_server_info,
        )

        # Create response - authentication should be applied
        response = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify operations succeeded
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None


def test_mcp_authentication_fallback_to_headers(compat_client, text_model_id):
    """Test that authentication parameter doesn't override existing headers."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    # Headers should take precedence - this test uses headers auth
    test_token = "headers-token-123"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "headers-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "headers": {"Authorization": f"Bearer {test_token}"},
                    "authentication": {
                        "type": "bearer",
                        "token": "should-not-override",
                    },
                }
            ],
            mcp_server_info,
        )

        # Create response - headers should take precedence
        response = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify operations succeeded with headers auth
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None


def test_mcp_authentication_backward_compatibility(compat_client, text_model_id):
    """Test that MCP tools work without authentication (backward compatibility)."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    # No authentication required
    with make_mcp_server(required_auth_token=None) as mcp_server_info:
        tools = setup_mcp_tools(
            [{"type": "mcp", "server_label": "noauth-mcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}],
            mcp_server_info,
        )

        # Create response without authentication
        response = compat_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify operations succeeded without auth
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None
