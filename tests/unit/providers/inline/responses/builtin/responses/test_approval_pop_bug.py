# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for the next_turn_messages.pop() bug in _separate_tool_calls().

When multiple MCP tool calls require approval in a single model turn,
pop() must only remove the assistant message once — not once per tool call.

See: https://github.com/llamastack/llama-stack/issues/5301
"""

from unittest.mock import AsyncMock, MagicMock

from llama_stack.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator
from llama_stack.providers.inline.responses.builtin.responses.types import ChatCompletionContext, ToolContext
from llama_stack_api.inference.models import (
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
)
from llama_stack_api.openai_responses import OpenAIResponseInputToolMCP


def _make_mcp_server(**kwargs) -> OpenAIResponseInputToolMCP:
    defaults = {"server_label": "test-server", "server_url": "http://localhost:9999/mcp"}
    defaults.update(kwargs)
    return OpenAIResponseInputToolMCP(**defaults)


def _make_tool_call(call_id: str, name: str, arguments: str = "{}") -> OpenAIChatCompletionToolCall:
    return OpenAIChatCompletionToolCall(
        id=call_id,
        function=OpenAIChatCompletionToolCallFunction(name=name, arguments=arguments),
    )


def _build_orchestrator(mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP]) -> StreamingResponseOrchestrator:
    mock_ctx = MagicMock(spec=ChatCompletionContext)
    mock_ctx.tool_context = MagicMock(spec=ToolContext)
    mock_ctx.tool_context.previous_tools = mcp_tool_to_server
    mock_ctx.model = "test-model"
    mock_ctx.messages = []
    mock_ctx.temperature = None
    mock_ctx.top_p = None
    mock_ctx.frequency_penalty = None
    mock_ctx.response_format = MagicMock()
    mock_ctx.tool_choice = None
    mock_ctx.response_tools = [
        MagicMock(type="mcp", name="get_weather"),
        MagicMock(type="mcp", name="get_time"),
        MagicMock(type="mcp", name="get_news"),
    ]
    mock_ctx.approval_response = MagicMock(return_value=None)

    return StreamingResponseOrchestrator(
        inference_api=AsyncMock(),
        ctx=mock_ctx,
        response_id="resp_test",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        safety_api=None,
    )


def _make_response(tool_calls: list[OpenAIChatCompletionToolCall]):
    """Build a mock chat completion response with a single choice containing the given tool calls."""
    return MagicMock(
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason="tool_calls",
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls,
                ),
            )
        ]
    )


class TestSeparateToolCallsPopBug:
    """Verify that next_turn_messages.pop() only fires once per choice."""

    def test_single_approval_pops_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [_make_tool_call("call_1", "get_weather")]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 1
        assert len(result_messages) == 2, "Only the original messages should remain (assistant popped)"
        assert result_messages == ["system_msg", "user_msg"]

    def test_multiple_approvals_pop_only_once(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server, "get_news": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
            _make_tool_call("call_3", "get_news"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 3
        assert len(result_messages) == 2, (
            f"Expected 2 messages (original preserved), got {len(result_messages)}. "
            "The pop() bug is removing more than the assistant message."
        )
        assert result_messages == ["system_msg", "user_msg"]

    def test_two_approvals_does_not_eat_user_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 2
        assert "user_msg" in result_messages, "User message must not be removed by pop()"
        assert "system_msg" in result_messages, "System message must not be removed by pop()"

    def test_denied_tool_calls_pop_only_once(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        denial = MagicMock()
        denial.approve = False
        orch.ctx.approval_response = MagicMock(return_value=denial)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 0
        assert len(result_messages) == 2
        assert result_messages == ["system_msg", "user_msg"]

    def test_mix_of_approved_and_pending_pops_once(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True

        def side_effect(name, args):
            if name == "get_weather":
                return approval
            return None

        orch.ctx.approval_response = MagicMock(side_effect=side_effect)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 1
        assert len(approvals) == 1
        assert len(result_messages) == 2
        assert result_messages == ["system_msg", "user_msg"]

    def test_no_approvals_needed_keeps_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="never")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 2
        assert len(approvals) == 0
        # Assistant message should remain since all tool calls are approved
        assert len(result_messages) == 3
