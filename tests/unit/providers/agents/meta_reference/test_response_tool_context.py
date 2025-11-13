# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.agents.openai_responses import (
    MCPListToolsTool,
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseToolMCP,
)
from llama_stack.providers.inline.agents.meta_reference.responses.types import ToolContext


class TestToolContext:
    def test_no_tools(self):
        tools = []
        context = ToolContext(tools)
        previous_response = OpenAIResponseObject(created_at=1234, id="test", model="mymodel", output=[], status="")
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 0
        assert len(context.previous_tools) == 0
        assert len(context.previous_tool_listings) == 0

    def test_no_previous_tools(self):
        tools = [
            OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
            OpenAIResponseInputToolMCP(server_label="label", server_url="url"),
        ]
        context = ToolContext(tools)
        previous_response = OpenAIResponseObject(created_at=1234, id="test", model="mymodel", output=[], status="")
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 2
        assert len(context.previous_tools) == 0
        assert len(context.previous_tool_listings) == 0

    def test_reusable_server(self):
        tools = [
            OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test", server_label="alabel", tools=[MCPListToolsTool(name="test_tool", input_schema={})]
            )
        ]
        previous_response = OpenAIResponseObject(created_at=1234, id="test", model="fake", output=output, status="")
        previous_response.tools = [
            OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
            OpenAIResponseToolMCP(server_label="alabel"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 1
        assert context.tools_to_process[0].type == "file_search"
        assert len(context.previous_tools) == 1
        assert context.previous_tools["test_tool"].server_label == "alabel"
        assert context.previous_tools["test_tool"].server_url == "aurl"
        assert len(context.previous_tool_listings) == 1
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "alabel"

    def test_multiple_reusable_servers(self):
        tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test1", server_label="alabel", tools=[MCPListToolsTool(name="test_tool", input_schema={})]
            ),
            OpenAIResponseOutputMessageMCPListTools(
                id="test2",
                server_label="anotherlabel",
                tools=[MCPListToolsTool(name="some_other_tool", input_schema={})],
            ),
        ]
        previous_response = OpenAIResponseObject(created_at=1234, id="test", model="fake", output=output, status="")
        previous_response.tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 2
        assert context.tools_to_process[0].type == "function"
        assert context.tools_to_process[1].type == "web_search"
        assert len(context.previous_tools) == 2
        assert context.previous_tools["test_tool"].server_label == "alabel"
        assert context.previous_tools["test_tool"].server_url == "aurl"
        assert context.previous_tools["some_other_tool"].server_label == "anotherlabel"
        assert context.previous_tools["some_other_tool"].server_url == "anotherurl"
        assert len(context.previous_tool_listings) == 2
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "alabel"
        assert len(context.previous_tool_listings[1].tools) == 1
        assert context.previous_tool_listings[1].server_label == "anotherlabel"

    def test_multiple_servers_only_one_reusable(self):
        tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test2",
                server_label="anotherlabel",
                tools=[MCPListToolsTool(name="some_other_tool", input_schema={})],
            )
        ]
        previous_response = OpenAIResponseObject(created_at=1234, id="test", model="fake", output=output, status="")
        previous_response.tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 3
        assert context.tools_to_process[0].type == "function"
        assert context.tools_to_process[1].type == "web_search"
        assert context.tools_to_process[2].type == "mcp"
        assert len(context.previous_tools) == 1
        assert context.previous_tools["some_other_tool"].server_label == "anotherlabel"
        assert context.previous_tools["some_other_tool"].server_url == "anotherurl"
        assert len(context.previous_tool_listings) == 1
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "anotherlabel"

    def test_mismatched_allowed_tools(self):
        tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl", allowed_tools=["test_tool_2"]),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test1", server_label="alabel", tools=[MCPListToolsTool(name="test_tool_1", input_schema={})]
            ),
            OpenAIResponseOutputMessageMCPListTools(
                id="test2",
                server_label="anotherlabel",
                tools=[MCPListToolsTool(name="some_other_tool", input_schema={})],
            ),
        ]
        previous_response = OpenAIResponseObject(created_at=1234, id="test", model="fake", output=output, status="")
        previous_response.tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 3
        assert context.tools_to_process[0].type == "function"
        assert context.tools_to_process[1].type == "web_search"
        assert context.tools_to_process[2].type == "mcp"
        assert len(context.previous_tools) == 1
        assert context.previous_tools["some_other_tool"].server_label == "anotherlabel"
        assert context.previous_tools["some_other_tool"].server_url == "anotherurl"
        assert len(context.previous_tool_listings) == 1
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "anotherlabel"
