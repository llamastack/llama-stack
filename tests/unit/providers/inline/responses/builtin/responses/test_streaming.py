# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.providers.inline.responses.builtin.responses.streaming import (
    StreamingResponseOrchestrator,
    convert_tooldef_to_chat_tool,
)
from llama_stack.providers.inline.responses.builtin.responses.types import ChatCompletionContext
from llama_stack_api import (
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIResponseInputToolMCP,
    ToolDef,
)


@pytest.fixture
def mock_safety_api():
    safety_api = AsyncMock()
    # Mock the routing table and shields list for guardrails lookup
    safety_api.routing_table = AsyncMock()
    shield = AsyncMock()
    shield.identifier = "llama-guard"
    shield.provider_resource_id = "llama-guard-model"
    safety_api.routing_table.list_shields.return_value = AsyncMock(data=[shield])
    # Mock run_moderation to return non-flagged result by default
    safety_api.run_moderation.return_value = AsyncMock(flagged=False)
    return safety_api


@pytest.fixture
def mock_inference_api():
    inference_api = AsyncMock()
    return inference_api


@pytest.fixture
def mock_context():
    context = AsyncMock(spec=ChatCompletionContext)
    # Add required attributes that StreamingResponseOrchestrator expects
    context.tool_context = AsyncMock()
    context.tool_context.previous_tools = {}
    context.messages = []
    return context


def test_convert_tooldef_to_chat_tool_preserves_items_field():
    """Test that array parameters preserve the items field during conversion.

    This test ensures that when converting ToolDef with array-type parameters
    to OpenAI ChatCompletionToolParam format, the 'items' field is preserved.
    Without this fix, array parameters would be missing schema information about their items.
    """
    tool_def = ToolDef(
        name="test_tool",
        description="A test tool with array parameter",
        input_schema={
            "type": "object",
            "properties": {"tags": {"type": "array", "description": "List of tags", "items": {"type": "string"}}},
            "required": ["tags"],
        },
    )

    result = convert_tooldef_to_chat_tool(tool_def)

    assert result["type"] == "function"
    assert result["function"]["name"] == "test_tool"

    tags_param = result["function"]["parameters"]["properties"]["tags"]
    assert tags_param["type"] == "array"
    assert "items" in tags_param, "items field should be preserved for array parameters"
    assert tags_param["items"] == {"type": "string"}


def _make_mcp_tool(server_label="mcp_server", require_approval="always"):
    return OpenAIResponseInputToolMCP(
        server_label=server_label,
        server_url="http://localhost:8080",
        require_approval=require_approval,
    )


def _make_tool_call(call_id, name, arguments="{}"):
    return OpenAIChatCompletionToolCall(
        id=call_id,
        type="function",
        function=OpenAIChatCompletionToolCallFunction(name=name, arguments=arguments),
    )


def _make_response(*choices):
    return SimpleNamespace(choices=list(choices))


def _make_choice(content, tool_calls):
    return OpenAIChoice(
        index=0,
        finish_reason="tool_calls",
        message=OpenAIChatCompletionResponseMessage(
            content=content,
            tool_calls=tool_calls,
        ),
    )


def _build_orchestrator(mcp_tool_mapping, approval_responses=None):
    """Build a minimal StreamingResponseOrchestrator with mocked dependencies."""
    ctx = MagicMock(spec=ChatCompletionContext)
    ctx.tool_context = MagicMock()
    ctx.tool_context.previous_tools = mcp_tool_mapping

    # response_tools must be non-empty for the tool classification branch to run
    ctx.response_tools = [MagicMock()]

    # approval_responses: dict mapping (tool_name) -> approval mock or None
    approval_map = approval_responses or {}

    def _approval_response(name, arguments):
        return approval_map.get(name)

    ctx.approval_response = _approval_response

    orchestrator = object.__new__(StreamingResponseOrchestrator)
    orchestrator.ctx = ctx
    orchestrator.mcp_tool_to_server = mcp_tool_mapping
    return orchestrator


class TestSeparateToolCalls:
    def test_single_approval_pending_removes_assistant_msg(self):
        """A single tool call needing approval should remove the assistant message."""
        mcp_tool = _make_mcp_tool()
        orchestrator = _build_orchestrator({"mcp_send": mcp_tool})

        tc = _make_tool_call("tc_1", "mcp_send")
        choice = _make_choice("I'll send a message", [tc])
        response = _make_response(choice)

        messages = [{"role": "user", "content": "send a msg"}]
        func_calls, non_func_calls, approvals, next_msgs = orchestrator._separate_tool_calls(response, messages)

        assert len(approvals) == 1
        assert len(next_msgs) == 1, "Only the original user message should remain"
        assert next_msgs[0] == messages[0]

    def test_multiple_approvals_pending_does_not_corrupt_history(self):
        """Multiple tool calls needing approval must only pop the assistant message once."""
        mcp_tool = _make_mcp_tool()
        orchestrator = _build_orchestrator({"mcp_send": mcp_tool, "mcp_delete": mcp_tool, "mcp_invite": mcp_tool})

        tc_a = _make_tool_call("tc_a", "mcp_send")
        tc_b = _make_tool_call("tc_b", "mcp_delete")
        tc_c = _make_tool_call("tc_c", "mcp_invite")
        choice = _make_choice("I'll do three things", [tc_a, tc_b, tc_c])
        response = _make_response(choice)

        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "previous reply"},
            {"role": "user", "content": "current request"},
        ]
        func_calls, non_func_calls, approvals, next_msgs = orchestrator._separate_tool_calls(response, messages)

        assert len(approvals) == 3
        assert len(next_msgs) == 3, "Original messages must be intact; only the new assistant message is removed"
        assert next_msgs == messages

    def test_multiple_approvals_denied_does_not_corrupt_history(self):
        """Multiple denied tool calls must only pop the assistant message once."""
        mcp_tool = _make_mcp_tool()
        denied = MagicMock()
        denied.approve = False
        orchestrator = _build_orchestrator(
            {"mcp_send": mcp_tool, "mcp_delete": mcp_tool},
            approval_responses={"mcp_send": denied, "mcp_delete": denied},
        )

        tc_a = _make_tool_call("tc_a", "mcp_send")
        tc_b = _make_tool_call("tc_b", "mcp_delete")
        choice = _make_choice("I'll do two things", [tc_a, tc_b])
        response = _make_response(choice)

        messages = [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": "old reply"},
        ]
        func_calls, non_func_calls, approvals, next_msgs = orchestrator._separate_tool_calls(response, messages)

        assert len(func_calls) == 0
        assert len(non_func_calls) == 0
        assert len(next_msgs) == 2, "Original messages must be intact"
        assert next_msgs == messages

    def test_mix_of_approved_and_pending_preserves_history(self):
        """One approved + one pending tool call: assistant msg stays (approved tool needs it)."""
        mcp_tool = _make_mcp_tool()
        approved = MagicMock()
        approved.approve = True
        orchestrator = _build_orchestrator(
            {"mcp_approved": mcp_tool, "mcp_pending": mcp_tool},
            approval_responses={"mcp_approved": approved},
        )

        tc_approved = _make_tool_call("tc_1", "mcp_approved")
        tc_pending = _make_tool_call("tc_2", "mcp_pending")
        choice = _make_choice("two calls", [tc_approved, tc_pending])
        response = _make_response(choice)

        messages = [{"role": "user", "content": "request"}]
        func_calls, non_func_calls, approvals, next_msgs = orchestrator._separate_tool_calls(response, messages)

        assert len(non_func_calls) == 1
        assert non_func_calls[0].id == "tc_1"
        assert len(approvals) == 1
        assert approvals[0].id == "tc_2"
        # The pending tool triggers removal; the approved tool was already classified
        assert len(next_msgs) == 1, "Assistant message should be removed due to pending approval"

    def test_no_approval_required_preserves_assistant_msg(self):
        """Tool calls that don't need approval should keep the assistant message."""
        mcp_tool = _make_mcp_tool(require_approval="never")
        orchestrator = _build_orchestrator({"web_search": mcp_tool})

        tc = _make_tool_call("tc_1", "web_search")
        choice = _make_choice("searching", [tc])
        response = _make_response(choice)

        messages = [{"role": "user", "content": "search for weather"}]
        func_calls, non_func_calls, approvals, next_msgs = orchestrator._separate_tool_calls(response, messages)

        assert len(non_func_calls) == 1
        assert len(approvals) == 0
        assert len(next_msgs) == 2, "User message + assistant message should both be present"
