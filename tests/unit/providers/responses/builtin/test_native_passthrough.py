# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionResult
from ogx_api import (
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone,
    OpenAIResponseObjectStreamResponseInProgress,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseObjectStreamResponseReasoningTextDelta,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseReasoning,
)
from ogx_api.openai_responses import (
    OpenAIResponseObject,
    OpenAIResponseUsage,
    OpenAIResponseUsageInputTokensDetails,
    OpenAIResponseUsageOutputTokensDetails,
)

# --- Fallback behavior ---


async def test_fallback_to_cc_when_native_not_supported(openai_responses_impl, mock_inference_api):
    """When check_native_responses_support returns False, the orchestrator
    should use openai_chat_completion instead of openai_response."""
    from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream

    mock_inference_api.check_native_responses_support = AsyncMock(return_value=False)
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    result = await openai_responses_impl.create_openai_response(
        input="Hello",
        model="meta-llama/Llama-3.1-8B-Instruct",
        stream=True,
    )
    chunks = [chunk async for chunk in result]

    mock_inference_api.openai_chat_completion.assert_called_once()
    mock_inference_api.openai_response.assert_not_called()
    assert any(c.type == "response.completed" for c in chunks)


async def test_native_path_used_when_supported(openai_responses_impl, mock_inference_api):
    """When check_native_responses_support returns True, CC should not be called."""
    mock_inference_api.check_native_responses_support = AsyncMock(return_value=True)
    response_obj = OpenAIResponseObject(
        id="resp_native",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_stream():
        yield OpenAIResponseObjectStreamResponseCreated(
            response=response_obj,
            sequence_number=0,
        )
        yield OpenAIResponseObjectStreamResponseInProgress(
            response=response_obj,
            sequence_number=1,
        )
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0,
            delta="Hello world",
            item_id="msg_1",
            output_index=0,
            sequence_number=2,
        )
        yield OpenAIResponseObjectStreamResponseCompleted(
            response=response_obj,
            sequence_number=3,
        )

    mock_inference_api.openai_response = AsyncMock(return_value=native_stream())

    result = await openai_responses_impl.create_openai_response(
        input="Hello",
        model="meta-llama/Llama-3.1-8B-Instruct",
        stream=True,
    )
    chunks = [chunk async for chunk in result]

    mock_inference_api.openai_chat_completion.assert_not_called()
    assert any(c.type == "response.completed" for c in chunks)


async def test_native_path_extracts_text_content(openai_responses_impl, mock_inference_api):
    """The native path should extract text deltas into the response output."""
    mock_inference_api.check_native_responses_support = AsyncMock(return_value=True)
    response_obj = OpenAIResponseObject(
        id="resp_text",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseInProgress(response=response_obj, sequence_number=1)
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="The capital ", item_id="msg_1", output_index=0, sequence_number=2
        )
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="is Paris.", item_id="msg_1", output_index=0, sequence_number=3
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=4)

    mock_inference_api.openai_response = AsyncMock(return_value=native_stream())

    result = await openai_responses_impl.create_openai_response(
        input="What is the capital of France?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        stream=True,
    )
    chunks = [chunk async for chunk in result]

    completed = [c for c in chunks if c.type == "response.completed"]
    assert len(completed) >= 1
    resp = completed[-1].response
    assert resp.status == "completed"
    output_messages = [o for o in resp.output if o.type == "message"]
    assert len(output_messages) == 1
    assert output_messages[0].content[0].text == "The capital is Paris."


async def test_native_path_extracts_tool_calls(openai_responses_impl, mock_inference_api):
    """The native path should extract tool calls from OutputItemDone events
    and include them in the ChatCompletionResult for the orchestrator's tool loop."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    response_obj = OpenAIResponseObject(
        id="resp_tools",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    tool_call_item = OpenAIResponseOutputMessageFunctionToolCall(
        id="fc_1",
        call_id="call_abc",
        name="get_weather",
        arguments='{"location": "Paris"}',
        type="function_call",
        status="completed",
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone(
            arguments='{"location": "Paris"}',
            item_id="fc_1",
            output_index=0,
            sequence_number=1,
        )
        yield OpenAIResponseObjectStreamResponseOutputItemDone(
            response_id="resp_tools",
            item=tool_call_item,
            output_index=0,
            sequence_number=2,
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=3)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.guardrail_ids = []
    orchestrator.safety_api = None
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0

    events = []
    result = None
    async for event_or_result in orchestrator._process_native_response_events(native_event_stream()):
        if isinstance(event_or_result, ChatCompletionResult):
            result = event_or_result
        else:
            events.append(event_or_result)

    assert result is not None
    assert result.has_tool_calls
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert "Paris" in tool_call.function.arguments
    assert tool_call.id == "call_abc"
    assert len(events) == 2


async def test_native_path_extracts_reasoning(openai_responses_impl, mock_inference_api):
    """The native path should extract reasoning text deltas."""
    mock_inference_api.check_native_responses_support = AsyncMock(return_value=True)
    response_obj = OpenAIResponseObject(
        id="resp_reasoning",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseInProgress(response=response_obj, sequence_number=1)
        yield OpenAIResponseObjectStreamResponseReasoningTextDelta(
            content_index=0, delta="Let me think...", item_id="rs_1", output_index=0, sequence_number=2
        )
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="The answer is 42.", item_id="msg_1", output_index=0, sequence_number=3
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=4)

    mock_inference_api.openai_response = AsyncMock(return_value=native_stream())

    result = await openai_responses_impl.create_openai_response(
        input="What is the meaning of life?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        stream=True,
        reasoning=OpenAIResponseReasoning(effort="high"),
    )
    chunks = [chunk async for chunk in result]

    completed = [c for c in chunks if c.type == "response.completed"]
    assert len(completed) >= 1
    resp = completed[-1].response
    assert resp.status == "completed"
    output_messages = [o for o in resp.output if o.type == "message"]
    assert len(output_messages) == 1
    assert output_messages[0].content[0].text == "The answer is 42."

    reasoning_items = [o for o in resp.output if o.type == "reasoning"]
    assert len(reasoning_items) == 1
    assert "think" in reasoning_items[0].content[0].text.lower()


async def test_native_path_incomplete_sets_finish_reason_length(openai_responses_impl, mock_inference_api):
    """When vLLM returns status=incomplete in the completed event,
    _process_native_response_events should set finish_reason=length in ChatCompletionResult."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    incomplete_response = OpenAIResponseObject(
        id="resp_inc",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="incomplete",
        store=True,
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=incomplete_response, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="Partial", item_id="msg_1", output_index=0, sequence_number=1
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=incomplete_response, sequence_number=2)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.guardrail_ids = []
    orchestrator.safety_api = None
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0

    events = []
    result = None
    async for event_or_result in orchestrator._process_native_response_events(native_event_stream()):
        if isinstance(event_or_result, ChatCompletionResult):
            result = event_or_result
        else:
            events.append(event_or_result)

    assert result is not None
    assert result.finish_reason == "length"
    assert result.content_text == "Partial"
    assert len(events) == 1


async def test_native_path_filters_lifecycle_events(openai_responses_impl, mock_inference_api):
    """Provider lifecycle events (created, in_progress, completed) should be filtered
    out since Llama Stack emits its own."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    response_obj = OpenAIResponseObject(
        id="resp_lifecycle",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseInProgress(response=response_obj, sequence_number=1)
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="Hello", item_id="msg_1", output_index=0, sequence_number=2
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=3)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.guardrail_ids = []
    orchestrator.safety_api = None
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0

    events = []
    result = None
    async for event_or_result in orchestrator._process_native_response_events(native_event_stream()):
        if isinstance(event_or_result, ChatCompletionResult):
            result = event_or_result
        else:
            events.append(event_or_result)

    assert result is not None
    assert len(events) == 1
    assert isinstance(events[0], OpenAIResponseObjectStreamResponseOutputTextDelta)
    assert events[0].delta == "Hello"


# --- Output guardrails ---


async def test_native_path_output_guardrail_blocks_unsafe_content(openai_responses_impl, mock_inference_api):
    """When output guardrails detect a violation, the native path should yield a
    refusal response and set violation_detected."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    response_obj = OpenAIResponseObject(
        id="resp_guardrail",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="unsafe content", item_id="msg_1", output_index=0, sequence_number=1
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=2)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.ctx.frequency_penalty = None
    orchestrator.ctx.temperature = None
    orchestrator.ctx.top_p = None
    orchestrator.ctx.tool_choice = None
    orchestrator.ctx.available_tools = lambda: []
    orchestrator.guardrail_ids = ["test-guardrail"]
    orchestrator.safety_api = AsyncMock()
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0
    orchestrator.sequence_number = 0
    orchestrator.top_logprobs = None
    orchestrator.truncation = None
    orchestrator.max_output_tokens = None
    orchestrator.safety_identifier = None
    orchestrator.service_tier = None
    orchestrator.metadata = None
    orchestrator.presence_penalty = None
    orchestrator.store = False
    orchestrator.prompt_cache_key = None

    with patch(
        "ogx.providers.inline.responses.builtin.responses.streaming.run_guardrails",
        new_callable=AsyncMock,
        return_value="Content blocked by safety policy",
    ):
        events = []
        async for event_or_result in orchestrator._process_native_response_events(native_event_stream()):
            if not isinstance(event_or_result, ChatCompletionResult):
                events.append(event_or_result)

    assert orchestrator.violation_detected is True
    assert len(events) == 1
    assert events[0].type == "response.completed"


async def test_native_path_output_guardrail_allows_safe_content(openai_responses_impl, mock_inference_api):
    """When output guardrails pass, events should be yielded normally."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    response_obj = OpenAIResponseObject(
        id="resp_safe",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="safe content", item_id="msg_1", output_index=0, sequence_number=1
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=2)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.guardrail_ids = ["test-guardrail"]
    orchestrator.safety_api = AsyncMock()
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0

    with patch(
        "ogx.providers.inline.responses.builtin.responses.streaming.run_guardrails",
        new_callable=AsyncMock,
        return_value=None,
    ):
        events = []
        result = None
        async for event_or_result in orchestrator._process_native_response_events(native_event_stream()):
            if isinstance(event_or_result, ChatCompletionResult):
                result = event_or_result
            else:
                events.append(event_or_result)

    assert orchestrator.violation_detected is False
    assert result is not None
    assert len(events) == 1
    assert events[0].delta == "safe content"


# --- Usage accumulation ---


async def test_native_path_accumulates_usage_from_completed_event(openai_responses_impl, mock_inference_api):
    """Usage from the completed response should be accumulated."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    usage = OpenAIResponseUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_tokens_details=OpenAIResponseUsageInputTokensDetails(cached_tokens=20),
        output_tokens_details=OpenAIResponseUsageOutputTokensDetails(reasoning_tokens=10),
    )
    response_obj = OpenAIResponseObject(
        id="resp_usage",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
        usage=usage,
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="Hello", item_id="msg_1", output_index=0, sequence_number=1
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=2)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.guardrail_ids = []
    orchestrator.safety_api = None
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0

    async for _ in orchestrator._process_native_response_events(native_event_stream()):
        pass

    assert orchestrator.accumulated_usage is not None
    assert orchestrator.accumulated_usage.input_tokens == 100
    assert orchestrator.accumulated_usage.output_tokens == 50
    assert orchestrator.accumulated_usage.total_tokens == 150
    assert orchestrator.accumulated_builtin_output_tokens == 50


# --- Reasoning deduplication ---


async def test_native_path_does_not_yield_reasoning_deltas(openai_responses_impl, mock_inference_api):
    """Reasoning text deltas should be accumulated but not yielded — the shared code
    in create_response handles emitting the ReasoningItem."""
    from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator

    response_obj = OpenAIResponseObject(
        id="resp_reasoning",
        created_at=1000,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )

    async def native_event_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=response_obj, sequence_number=0)
        yield OpenAIResponseObjectStreamResponseReasoningTextDelta(
            content_index=0, delta="Let me think...", item_id="rs_1", output_index=0, sequence_number=1
        )
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0, delta="The answer.", item_id="msg_1", output_index=0, sequence_number=2
        )
        yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj, sequence_number=3)

    orchestrator = StreamingResponseOrchestrator.__new__(StreamingResponseOrchestrator)
    orchestrator.response_id = "resp_test"
    orchestrator.created_at = 1000
    orchestrator.ctx = AsyncMock()
    orchestrator.ctx.model = "test-model"
    orchestrator.guardrail_ids = []
    orchestrator.safety_api = None
    orchestrator.violation_detected = False
    orchestrator.accumulated_usage = None
    orchestrator.accumulated_builtin_output_tokens = 0

    events = []
    result = None
    async for event_or_result in orchestrator._process_native_response_events(native_event_stream()):
        if isinstance(event_or_result, ChatCompletionResult):
            result = event_or_result
        else:
            events.append(event_or_result)

    assert result is not None
    assert result.reasoning_content == "Let me think..."
    assert len(events) == 1
    assert isinstance(events[0], OpenAIResponseObjectStreamResponseOutputTextDelta)
    assert events[0].delta == "The answer."
