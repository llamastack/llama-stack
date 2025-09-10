# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import pytest
from llama_stack_client import Agent

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.core.datatypes import AuthenticationRequiredError

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
        llama_stack_client.tools.list()

    response = llama_stack_client.tools.list(
        toolgroup_id=test_toolgroup_id,
        extra_headers=auth_headers,
    )
    assert len(response) == 2
    assert {t.identifier for t in response} == {"greet_everyone", "get_boiling_point"}

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
    agent = Agent(
        client=llama_stack_client,
        model=text_model_id,
        instructions="You are a helpful assistant.",
        tools=[test_toolgroup_id],
    )
    session_id = agent.create_session("test-session")
    response = agent.create_turn(
        session_id=session_id,
        messages=[
            {
                "role": "user",
                "content": "Say hi to the world. Use tools to do so.",
            }
        ],
        stream=False,
        extra_headers=auth_headers,
    )
    steps = response.steps
    first = steps[0]
    assert first.step_type == "inference"
    assert len(first.api_model_response.tool_calls) == 1
    tool_call = first.api_model_response.tool_calls[0]
    assert tool_call.tool_name == "greet_everyone"

    second = steps[1]
    assert second.step_type == "tool_execution"
    tool_response_content = second.tool_responses[0].content
    assert len(tool_response_content) == 1
    assert tool_response_content[0].type == "text"
    assert tool_response_content[0].text == "Hello, world!"

    third = steps[2]
    assert third.step_type == "inference"

    # when streaming, we currently don't check auth headers upfront and fail the request
    # early. but we should at least be generating a 401 later in the process.
    response = agent.create_turn(
        session_id=session_id,
        messages=[
            {
                "role": "user",
                "content": "What is the boiling point of polyjuice? Use tools to answer.",
            }
        ],
        stream=True,
    )
    if isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        with pytest.raises(AuthenticationRequiredError):
            for _ in response:
                pass
    else:
        error_chunks = [chunk for chunk in response if "error" in chunk.model_dump()]
        assert len(error_chunks) == 1
        chunk = error_chunks[0].model_dump()
        assert "Unauthorized" in chunk["error"]["message"]
