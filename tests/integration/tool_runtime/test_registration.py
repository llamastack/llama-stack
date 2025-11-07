# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re

import pytest

from llama_stack.apis.common.errors import ToolGroupNotFoundError
from llama_stack.core.library_client import LlamaStackAsLibraryClient
from tests.common.mcp import MCP_TOOLGROUP_ID, make_mcp_server


def test_register_and_unregister_toolgroup(llama_stack_client):
    # TODO: make this work for http client also but you need to ensure
    # the MCP server is reachable from llama stack server
    if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        pytest.skip("The local MCP server only reliably reachable from library client.")

    test_toolgroup_id = MCP_TOOLGROUP_ID
    provider_id = "model-context-protocol"

    with make_mcp_server() as mcp_server_info:
        # Cleanup before running the test
        toolgroups = llama_stack_client.toolgroups.list()
        for toolgroup in toolgroups:
            if toolgroup.identifier == test_toolgroup_id:
                llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)

        # Register the toolgroup
        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id=provider_id,
            mcp_endpoint=dict(uri=mcp_server_info["server_url"]),
        )

        # Verify registration
        registered_toolgroup = llama_stack_client.toolgroups.get(toolgroup_id=test_toolgroup_id)
        assert registered_toolgroup is not None
        assert registered_toolgroup.identifier == test_toolgroup_id
        assert registered_toolgroup.provider_id == provider_id

        # Verify tools listing
        tools_list_response = llama_stack_client.tools.list(toolgroup_id=test_toolgroup_id)
        assert isinstance(tools_list_response, list)
        assert tools_list_response

        # Unregister the toolgroup
        llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)

        # Verify it is unregistered
        with pytest.raises(
            ToolGroupNotFoundError,
            match=re.escape(
                f"Tool Group '{test_toolgroup_id}' not found. Use 'client.toolgroups.list()' to list available Tool Groups."
            ),
        ):
            llama_stack_client.toolgroups.get(toolgroup_id=test_toolgroup_id)

        with pytest.raises(
            ToolGroupNotFoundError,
            match=re.escape(
                f"Tool Group '{test_toolgroup_id}' not found. Use 'client.toolgroups.list()' to list available Tool Groups."
            ),
        ):
            llama_stack_client.tools.list(toolgroup_id=test_toolgroup_id)
