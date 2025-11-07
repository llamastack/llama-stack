# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.core.library_client import LlamaStackAsLibraryClient
from tests.common.mcp import make_mcp_server

from .helpers import setup_mcp_tools

# Skip these tests in replay mode until recordings are generated
pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMA_STACK_TEST_INFERENCE_MODE") == "replay",
    reason="No recordings yet for authorization tests. Run with --inference-mode=record-if-missing to generate.",
)


def test_mcp_authorization_bearer(compat_client, text_model_id):
    """Test that bearer authorization is correctly applied to MCP requests."""
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
                    "authorization": test_token,  # Just the token, not "Bearer <token>"
                }
            ],
            mcp_server_info,
        )

        # Create response - authorization should be applied
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


def test_mcp_authorization_different_token(compat_client, text_model_id):
    """Test authorization with a different bearer token."""
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
                    "authorization": test_token,
                }
            ],
            mcp_server_info,
        )

        # Create response - authorization should be applied
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


def test_mcp_authorization_error_when_header_provided(compat_client, text_model_id):
    """Test that providing Authorization in headers raises a security error."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    test_token = "test-token-123"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "header-auth-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "headers": {
                        "Authorization": f"Bearer {test_token}"
                    },  # Security risk - should be rejected
                }
            ],
            mcp_server_info,
        )

        # Create response - should raise ValueError for security reasons
        with pytest.raises(
            ValueError, match="Authorization header cannot be passed via 'headers'"
        ):
            compat_client.responses.create(
                model=text_model_id,
                input="What is the boiling point of myawesomeliquid?",
                tools=tools,
                stream=False,
            )


def test_mcp_authorization_backward_compatibility(compat_client, text_model_id):
    """Test that MCP tools work without authorization (backward compatibility)."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    # No authorization required
    with make_mcp_server(required_auth_token=None) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "noauth-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                }
            ],
            mcp_server_info,
        )

        # Create response without authorization
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
