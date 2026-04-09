# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.providers.inline.responses.builtin.responses.streaming import (
    StreamingResponseOrchestrator,
    convert_tooldef_to_chat_tool,
)
from llama_stack.providers.inline.responses.builtin.responses.types import ChatCompletionContext, ToolContext
from llama_stack.providers.inline.responses.builtin.responses.utils import (
    build_summary_prompt,
    should_summarize_reasoning,
    summarize_reasoning,
)
from llama_stack_api import ToolDef
from llama_stack_api.inference.models import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
)
from llama_stack_api.openai_responses import (
    OpenAIResponseInputToolMCP,
    OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded,
    OpenAIResponseObjectStreamResponseReasoningSummaryPartDone,
    OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta,
    OpenAIResponseObjectStreamResponseReasoningSummaryTextDone,
    OpenAIResponseReasoning,
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


# ---------------------------------------------------------------------------
# _separate_tool_calls regression tests
# See: https://github.com/llamastack/llama-stack/issues/5301
# ---------------------------------------------------------------------------


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


class TestAllDeferredOrDenied:
    """When all tool calls are deferred/denied, the assistant message should be fully popped."""

    def test_single_approval_pops_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [_make_tool_call("call_1", "get_weather")]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 1
        assert len(result_messages) == 2
        assert result_messages == ["system_msg", "user_msg"]

    def test_multiple_approvals_pops_once_not_per_tool_call(self):
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
        assert "user_msg" in result_messages
        assert "system_msg" in result_messages

    def test_all_denied_pops_assistant_message(self):
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


class TestMixedApproval:
    """When some tool calls are executed and some deferred/denied, the assistant
    message should be replaced with one containing only the executed tool calls."""

    def test_mix_replaces_assistant_message_with_executed_only(self):
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

        tc_weather = _make_tool_call("call_1", "get_weather")
        tc_time = _make_tool_call("call_2", "get_time")
        response = _make_response([tc_weather, tc_time])
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 1
        assert non_function[0].id == "call_1"
        assert len(approvals) == 1
        assert approvals[0].id == "call_2"

        assert len(result_messages) == 3
        assert result_messages[0] == "system_msg"
        assert result_messages[1] == "user_msg"

        replaced_msg = result_messages[2]
        assert isinstance(replaced_msg, OpenAIAssistantMessageParam)
        assert len(replaced_msg.tool_calls) == 1
        assert replaced_msg.tool_calls[0].id == "call_1"

    def test_mix_with_two_executed_one_deferred(self):
        always_server = _make_mcp_server(require_approval="always")
        never_server = _make_mcp_server(require_approval="never")
        tool_map = {"get_weather": never_server, "get_time": never_server, "get_news": always_server}
        orch = _build_orchestrator(tool_map)

        tc_weather = _make_tool_call("call_1", "get_weather")
        tc_time = _make_tool_call("call_2", "get_time")
        tc_news = _make_tool_call("call_3", "get_news")
        response = _make_response([tc_weather, tc_time, tc_news])
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 2
        assert len(approvals) == 1
        assert approvals[0].id == "call_3"

        replaced_msg = result_messages[2]
        assert isinstance(replaced_msg, OpenAIAssistantMessageParam)
        assert len(replaced_msg.tool_calls) == 2
        tool_call_ids = {tc.id for tc in replaced_msg.tool_calls}
        assert tool_call_ids == {"call_1", "call_2"}

    def test_mix_denied_and_executed_replaces_correctly(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True
        denial = MagicMock()
        denial.approve = False

        def side_effect(name, args):
            if name == "get_weather":
                return approval
            return denial

        orch.ctx.approval_response = MagicMock(side_effect=side_effect)

        tc_weather = _make_tool_call("call_1", "get_weather")
        tc_time = _make_tool_call("call_2", "get_time")
        response = _make_response([tc_weather, tc_time])
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 1
        assert len(approvals) == 0

        replaced_msg = result_messages[2]
        assert isinstance(replaced_msg, OpenAIAssistantMessageParam)
        assert len(replaced_msg.tool_calls) == 1
        assert replaced_msg.tool_calls[0].id == "call_1"

    def test_original_messages_always_preserved(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server, "get_news": mcp_server}
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
            _make_tool_call("call_3", "get_news"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, _, result_messages = orch._separate_tool_calls(response, messages)

        assert result_messages[0] == "system_msg"
        assert result_messages[1] == "user_msg"


class TestAllExecuted:
    """When all tool calls are executed, the assistant message should remain untouched."""

    def test_no_approvals_needed_keeps_full_assistant_message(self):
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
        assert len(result_messages) == 3

        assistant_msg = result_messages[2]
        assert isinstance(assistant_msg, OpenAIAssistantMessageParam)
        assert len(assistant_msg.tool_calls) == 2

    def test_all_pre_approved_keeps_full_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True
        orch.ctx.approval_response = MagicMock(return_value=approval)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 2
        assert len(approvals) == 0
        assert len(result_messages) == 3

        assistant_msg = result_messages[2]
        assert isinstance(assistant_msg, OpenAIAssistantMessageParam)
        assert len(assistant_msg.tool_calls) == 2


# ---------------------------------------------------------------------------
# Reasoning summary tests
# ---------------------------------------------------------------------------


class TestShouldSummarizeReasoning:
    def test_returns_false_when_reasoning_is_none(self):
        assert should_summarize_reasoning(None) is False

    def test_returns_true_for_concise(self):
        reasoning = OpenAIResponseReasoning(summary="concise")
        assert should_summarize_reasoning(reasoning) is True

    def test_returns_true_for_detailed(self):
        reasoning = OpenAIResponseReasoning(summary="detailed")
        assert should_summarize_reasoning(reasoning) is True

    def test_returns_true_for_auto(self):
        reasoning = OpenAIResponseReasoning(summary="auto")
        assert should_summarize_reasoning(reasoning) is True


class TestBuildSummaryPrompt:
    def test_concise_prompt_asks_for_short_summary(self):
        prompt = build_summary_prompt("Some reasoning text", "concise")
        assert "one or two sentences" in prompt
        assert "Some reasoning text" in prompt

    def test_detailed_prompt_preserves_logical_steps(self):
        prompt = build_summary_prompt("Some reasoning text", "detailed")
        assert "Preserve the key logical steps" in prompt
        assert "Some reasoning text" in prompt

    def test_auto_falls_through_to_concise(self):
        prompt_auto = build_summary_prompt("text", "auto")
        prompt_concise = build_summary_prompt("text", "concise")
        assert prompt_auto == prompt_concise


def _make_streaming_chunk(content: str) -> OpenAIChatCompletionChunk:
    """Build a mock streaming chunk with a single delta containing content."""
    return OpenAIChatCompletionChunk(
        id="chunk_1",
        choices=[
            OpenAIChunkChoice(
                index=0,
                delta=OpenAIChoiceDelta(content=content),
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion.chunk",
    )


async def _to_async_iter(items):
    """Convert a list into an async iterator."""
    for item in items:
        yield item


class TestSummarizeReasoning:
    async def test_emits_correct_event_sequence_single_part(self):
        """Single-paragraph summary: PartAdded -> TextDelta -> TextDone -> PartDone."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("The answer"),
                _make_streaming_chunk(" is 4."),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="Simple math.",
            reasoning_item_id="rs_test123",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=10,
        ):
            events.append(event)

        assert len(events) == 4
        assert isinstance(events[0], OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded)
        assert events[0].summary_index == 0
        assert events[0].part.text == "", "PartAdded should carry an empty text placeholder"
        assert isinstance(events[1], OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta)
        assert isinstance(events[2], OpenAIResponseObjectStreamResponseReasoningSummaryTextDone)
        assert isinstance(events[3], OpenAIResponseObjectStreamResponseReasoningSummaryPartDone)

    async def test_emits_multiple_summary_parts(self):
        """Multi-paragraph summary should produce multiple PartAdded/TextDelta/TextDone/PartDone blocks."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("First paragraph."),
                _make_streaming_chunk("\n\nSecond paragraph."),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="complex reasoning",
            reasoning_item_id="rs_multi",
            output_index=0,
            summary_mode="detailed",
            start_sequence_number=0,
        ):
            events.append(event)

        # Two paragraphs -> 2 * (PartAdded + TextDelta + TextDone + PartDone) = 8 events
        assert len(events) == 8

        # First part (summary_index=0)
        assert isinstance(events[0], OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded)
        assert events[0].summary_index == 0
        assert isinstance(events[1], OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta)
        assert events[1].delta == "First paragraph."
        assert isinstance(events[2], OpenAIResponseObjectStreamResponseReasoningSummaryTextDone)
        assert events[2].text == "First paragraph."
        assert isinstance(events[3], OpenAIResponseObjectStreamResponseReasoningSummaryPartDone)
        assert events[3].part.text == "First paragraph."

        # Second part (summary_index=1)
        assert isinstance(events[4], OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded)
        assert events[4].summary_index == 1
        assert isinstance(events[5], OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta)
        assert events[5].delta == "Second paragraph."
        assert isinstance(events[6], OpenAIResponseObjectStreamResponseReasoningSummaryTextDone)
        assert events[6].text == "Second paragraph."
        assert isinstance(events[7], OpenAIResponseObjectStreamResponseReasoningSummaryPartDone)
        assert events[7].part.text == "Second paragraph."

    async def test_accumulates_text_in_single_part(self):
        """Single-paragraph: text_done and part_done should contain the full concatenated summary."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("Hello"),
                _make_streaming_chunk(" world"),
                _make_streaming_chunk("!"),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_abc",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=0,
        ):
            events.append(event)

        text_done = [e for e in events if isinstance(e, OpenAIResponseObjectStreamResponseReasoningSummaryTextDone)]
        assert len(text_done) == 1
        assert text_done[0].text == "Hello world!"

        part_done = [e for e in events if isinstance(e, OpenAIResponseObjectStreamResponseReasoningSummaryPartDone)]
        assert len(part_done) == 1
        assert part_done[0].part.text == "Hello world!"

    async def test_sequence_numbers_increment(self):
        """Each event should have a strictly increasing sequence_number."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("AB"),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_seq",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=5,
        ):
            events.append(event)

        # Single paragraph: PartAdded, TextDelta, TextDone, PartDone
        seq_numbers = [e.sequence_number for e in events]
        assert seq_numbers == [6, 7, 8, 9]

    async def test_sequence_numbers_increment_multi_part(self):
        """Sequence numbers should be strictly increasing across multiple summary parts."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("P1\n\nP2\n\nP3"),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_seq_multi",
            output_index=0,
            summary_mode="detailed",
            start_sequence_number=0,
        ):
            events.append(event)

        seq_numbers = [e.sequence_number for e in events]
        for i in range(1, len(seq_numbers)):
            assert seq_numbers[i] > seq_numbers[i - 1], f"Sequence numbers must be strictly increasing: {seq_numbers}"

    async def test_propagates_item_id_and_output_index(self):
        """All events should carry the correct item_id and output_index."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("summary"),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_myid",
            output_index=3,
            summary_mode="detailed",
            start_sequence_number=0,
        ):
            events.append(event)

        for event in events:
            assert event.item_id == "rs_myid"
            assert event.output_index == 3

    async def test_inference_failure_raises(self):
        """If the inference call raises, the error should propagate to the caller."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.side_effect = RuntimeError("provider down")

        with pytest.raises(RuntimeError, match="provider down"):
            async for _ in summarize_reasoning(
                inference_api=mock_inference,
                model="test-model",
                reasoning_text="reasoning",
                reasoning_item_id="rs_fail",
                output_index=0,
                summary_mode="concise",
                start_sequence_number=0,
            ):
                pass

    async def test_empty_stream_yields_nothing(self):
        """If the provider returns empty chunks, no events should be yielded."""
        mock_inference = AsyncMock()
        empty_chunk = OpenAIChatCompletionChunk(
            id="chunk_empty",
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(content=None),
                )
            ],
            created=0,
            model="test-model",
            object="chat.completion.chunk",
        )
        mock_inference.openai_chat_completion.return_value = _to_async_iter([empty_chunk])

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_empty",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=0,
        ):
            events.append(event)

        assert events == []

    async def test_non_streaming_response_yields_nothing(self):
        """If the inference API returns a non-streaming response, no events should be yielded."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = MagicMock()  # not an AsyncIterator

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_nonstream",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=0,
        ):
            events.append(event)

        assert events == []

    async def test_usage_chunks_collected(self):
        """Chunks with usage data should be appended to the usage_chunks list."""
        usage_data = OpenAIChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        chunk_with_usage = OpenAIChatCompletionChunk(
            id="chunk_u",
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(content="summary"),
                )
            ],
            created=0,
            model="test-model",
            object="chat.completion.chunk",
            usage=usage_data,
        )

        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter([chunk_with_usage])

        usage_chunks: list = []
        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_usage",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=0,
            usage_chunks=usage_chunks,
        ):
            events.append(event)

        assert len(usage_chunks) == 1
        assert usage_chunks[0].usage.prompt_tokens == 10
        assert usage_chunks[0].usage.completion_tokens == 5

    async def test_whitespace_only_paragraphs_ignored(self):
        """Paragraphs that are only whitespace should be filtered out."""
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _to_async_iter(
            [
                _make_streaming_chunk("Content\n\n  \n\nMore content"),
            ]
        )

        events = []
        async for event in summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            reasoning_item_id="rs_ws",
            output_index=0,
            summary_mode="concise",
            start_sequence_number=0,
        ):
            events.append(event)

        part_added = [e for e in events if isinstance(e, OpenAIResponseObjectStreamResponseReasoningSummaryPartAdded)]
        assert len(part_added) == 2
        text_deltas = [e for e in events if isinstance(e, OpenAIResponseObjectStreamResponseReasoningSummaryTextDelta)]
        assert text_deltas[0].delta == "Content"
        assert text_deltas[1].delta == "More content"
