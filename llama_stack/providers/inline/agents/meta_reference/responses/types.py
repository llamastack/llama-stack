# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseObjectStream,
    OpenAIResponseOutput,
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


class ChatCompletionContext(BaseModel):
    model: str
    messages: list[OpenAIMessageParam]
    response_tools: list[OpenAIResponseInputTool] | None = None
    chat_tools: list[ChatCompletionToolParam] | None = None
    temperature: float | None
    response_format: OpenAIResponseFormatParam
    approval_requests: list[OpenAIResponseMCPApprovalRequest] = []
    approval_responses: dict[str, OpenAIResponseMCPApprovalResponse] = {}

    def __init__(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        response_tools: list[OpenAIResponseInputTool] | None,
        temperature: float | None,
        response_format: OpenAIResponseFormatParam,
        inputs: list[OpenAIResponseInput] | str,
    ):
        super().__init__(
            model=model,
            messages=messages,
            response_tools=response_tools,
            temperature=temperature,
            response_format=response_format,
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
