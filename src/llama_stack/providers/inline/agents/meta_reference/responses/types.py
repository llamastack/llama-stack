# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass
from typing import cast

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseTool,
    OpenAIResponseToolMCP,
)
from llama_stack.apis.inference import OpenAIChatCompletionToolCall, OpenAIMessageParam, OpenAIResponseFormatParam


class ToolExecutionResult(BaseModel):
    """Result of streaming tool execution."""

    stream_event: OpenAIResponseObjectStream | None = None
    sequence_number: int
    final_output_message: OpenAIResponseOutput | None = None
    final_input_message: OpenAIMessageParam | None = None
    citation_files: dict[str, str] | None = None


@dataclass
class ChatCompletionResult:
    """Result of processing streaming chat completion chunks."""

    response_id: str
    content: list[str]
    tool_calls: dict[int, OpenAIChatCompletionToolCall]
    created: int
    model: str
    finish_reason: str
    message_item_id: str  # For streaming events
    tool_call_item_ids: dict[int, str]  # For streaming events
    content_part_emitted: bool  # Tracking state

    @property
    def content_text(self) -> str:
        """Get joined content as string."""
        return "".join(self.content)

    @property
    def has_tool_calls(self) -> bool:
        """Check if there are any tool calls."""
        return bool(self.tool_calls)


class ToolContext(BaseModel):
    """Holds information about tools from this and (if relevant)
    previous response in order to facilitate reuse of previous
    listings where appropriate."""

    # tools argument passed into current request:
    current_tools: list[OpenAIResponseInputTool]
    # reconstructed map of tool -> mcp server from previous response:
    previous_tools: dict[str, OpenAIResponseInputToolMCP]
    # reusable mcp-list-tools objects from previous response:
    previous_tool_listings: list[OpenAIResponseOutputMessageMCPListTools]
    # tool arguments from current request that still need to be processed:
    tools_to_process: list[OpenAIResponseInputTool]

    def __init__(
        self,
        current_tools: list[OpenAIResponseInputTool] | None,
    ):
        super().__init__(
            current_tools=current_tools or [],
            previous_tools={},
            previous_tool_listings=[],
            tools_to_process=current_tools or [],
        )

    def recover_tools_from_previous_response(
        self,
        previous_response: OpenAIResponseObject,
    ):
        """Determine which mcp_list_tools objects from previous response we can reuse."""

        if self.current_tools and previous_response.tools:
            previous_tools_by_label: dict[str, OpenAIResponseToolMCP] = {}
            for tool in previous_response.tools:
                if isinstance(tool, OpenAIResponseToolMCP):
                    previous_tools_by_label[tool.server_label] = tool
            # collect tool definitions which are the same in current and previous requests:
            tools_to_process: list[OpenAIResponseInputTool] = []
            matched: dict[str, OpenAIResponseInputToolMCP] = {}
            # Mypy confuses OpenAIResponseInputTool (Input union) with OpenAIResponseTool (output union)
            # which differ only in MCP type (InputToolMCP vs ToolMCP). Code is correct.
            for tool in cast(list[OpenAIResponseInputTool], self.current_tools):  # type: ignore[assignment]
                if isinstance(tool, OpenAIResponseInputToolMCP) and tool.server_label in previous_tools_by_label:
                    previous_tool = previous_tools_by_label[tool.server_label]
                    if previous_tool.allowed_tools == tool.allowed_tools:
                        matched[tool.server_label] = tool
                    else:
                        tools_to_process.append(tool)  # type: ignore[arg-type]
                else:
                    tools_to_process.append(tool)  # type: ignore[arg-type]
            # tools that are not the same or were not previously defined need to be processed:
            self.tools_to_process = tools_to_process
            # for all matched definitions, get the mcp_list_tools objects from the previous output:
            self.previous_tool_listings = [
                obj for obj in previous_response.output if obj.type == "mcp_list_tools" and obj.server_label in matched
            ]
            # reconstruct the tool to server mappings that can be reused:
            for listing in self.previous_tool_listings:
                # listing is OpenAIResponseOutputMessageMCPListTools which has tools: list[MCPListToolsTool]
                definition = matched[listing.server_label]
                for mcp_tool in listing.tools:
                    # mcp_tool is MCPListToolsTool which has a name: str field
                    self.previous_tools[mcp_tool.name] = definition

    def available_tools(self) -> list[OpenAIResponseTool]:
        if not self.current_tools:
            return []

        def convert_tool(tool: OpenAIResponseInputTool) -> OpenAIResponseTool:
            if isinstance(tool, OpenAIResponseInputToolWebSearch):
                return tool
            if isinstance(tool, OpenAIResponseInputToolFileSearch):
                return tool
            if isinstance(tool, OpenAIResponseInputToolFunction):
                return tool
            if isinstance(tool, OpenAIResponseInputToolMCP):
                return OpenAIResponseToolMCP(
                    server_label=tool.server_label,
                    allowed_tools=tool.allowed_tools,
                )
            # Exhaustive check - all tool types should be handled above
            raise AssertionError(f"Unexpected tool type: {type(tool)}")

        return [convert_tool(tool) for tool in self.current_tools]


class ChatCompletionContext(BaseModel):
    model: str
    messages: list[OpenAIMessageParam]
    response_tools: list[OpenAIResponseInputTool] | None = None
    chat_tools: list[ChatCompletionToolParam] | None = None
    temperature: float | None
    response_format: OpenAIResponseFormatParam
    tool_context: ToolContext | None
    approval_requests: list[OpenAIResponseMCPApprovalRequest] = []
    approval_responses: dict[str, OpenAIResponseMCPApprovalResponse] = {}

    def __init__(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        response_tools: list[OpenAIResponseInputTool] | None,
        temperature: float | None,
        response_format: OpenAIResponseFormatParam,
        tool_context: ToolContext,
        inputs: list[OpenAIResponseInput] | str,
    ):
        super().__init__(
            model=model,
            messages=messages,
            response_tools=response_tools,
            temperature=temperature,
            response_format=response_format,
            tool_context=tool_context,
        )
        if not isinstance(inputs, str):
            self.approval_requests = [input for input in inputs if input.type == "mcp_approval_request"]
            self.approval_responses = {
                input.approval_request_id: input for input in inputs if input.type == "mcp_approval_response"
            }

    def approval_response(self, tool_name: str, arguments: str) -> OpenAIResponseMCPApprovalResponse | None:
        request = self._approval_request(tool_name, arguments)
        return self.approval_responses.get(request.id, None) if request else None

    def _approval_request(self, tool_name: str, arguments: str) -> OpenAIResponseMCPApprovalRequest | None:
        for request in self.approval_requests:
            if request.name == tool_name and request.arguments == arguments:
                return request
        return None

    def available_tools(self) -> list[OpenAIResponseTool]:
        if not self.tool_context:
            return []
        return self.tool_context.available_tools()
