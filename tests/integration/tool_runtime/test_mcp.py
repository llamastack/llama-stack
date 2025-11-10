# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import pytest
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.turn_events import StepCompleted, StepProgress, ToolCallIssuedDelta

from llama_stack.core.library_client import LlamaStackAsLibraryClient

AUTH_TOKEN = "test-token"

from tests.common.mcp import MCP_TOOLGROUP_ID, make_mcp_server


@pytest.fixture(scope="function")
def mcp_server():
    with make_mcp_server(required_auth_token=AUTH_TOKEN) as mcp_server_info:
        yield mcp_server_info


def test_mcp_invocation(llama_stack_client, text_model_id, mcp_server):
    if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        pytest.skip("The local MCP server only reliably reachable from library client.")

    test_toolgroup_id = MCP_TOOLGROUP_ID
    uri = mcp_server["server_url"]

    # registering should not raise an error anymore even if you don't specify the auth token
    try:
        llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
    except Exception:
        pass

    llama_stack_client.toolgroups.register(
        toolgroup_id=test_toolgroup_id,
        provider_id="model-context-protocol",
        mcp_endpoint=dict(uri=uri),
    )

    provider_data = {
        "mcp_headers": {
            uri: {
                "Authorization": f"Bearer {AUTH_TOKEN}",
            },
        },
    }
    auth_headers = {
        "X-LlamaStack-Provider-Data": json.dumps(provider_data),
    }

    with pytest.raises(Exception, match="Unauthorized"):
        llama_stack_client.tools.list(toolgroup_id=test_toolgroup_id)

    tools_list = llama_stack_client.tools.list(
        toolgroup_id=test_toolgroup_id,
        extra_headers=auth_headers,
    )
    assert len(tools_list) == 2
    assert {t.name for t in tools_list} == {"greet_everyone", "get_boiling_point"}

    response = llama_stack_client.tool_runtime.invoke_tool(
        tool_name="greet_everyone",
        kwargs=dict(url="https://www.google.com"),
        extra_headers=auth_headers,
    )
    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert content[0].text == "Hello, world!"

    print(f"Using model: {text_model_id}")
    tool_defs = [
        {
            "type": "mcp",
            "server_url": uri,
            "server_label": test_toolgroup_id,
            "require_approval": "never",
            "allowed_tools": [tool.name for tool in tools_list],
            "headers": {
                "Authorization": f"Bearer {AUTH_TOKEN}",
            },
        }
    ]
    agent = Agent(
        client=llama_stack_client,
        model=text_model_id,
        instructions="You are a helpful assistant.",
        tools=tool_defs,
    )
    session_id = agent.create_session("test-session")
    chunks = list(
        agent.create_turn(
            session_id=session_id,
            messages=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Say hi to the world. Use tools to do so.",
                        }
                    ],
                }
            ],
            stream=True,
            extra_headers=auth_headers,
        )
    )
    events = [chunk.event for chunk in chunks]

    final_response = next((chunk.response for chunk in reversed(chunks) if chunk.response), None)
    assert final_response is not None

    issued_calls = [
        event for event in events if isinstance(event, StepProgress) and isinstance(event.delta, ToolCallIssuedDelta)
    ]
    assert issued_calls

    assert issued_calls[-1].delta.tool_name == "greet_everyone"

    tool_events = [
        event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
    ]
    assert tool_events
    assert tool_events[-1].result.tool_calls[0].tool_name == "greet_everyone"

    assert "hello" in final_response.output_text.lower()
