# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for background parameter support in Responses API."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from llama_stack.core.task import capture_otel_context, create_task_with_detached_otel_context
from llama_stack.providers.inline.agents.builtin.responses.openai_responses import OpenAIResponsesImpl
from llama_stack_api import OpenAIResponseError, OpenAIResponseObject


class _CollectingExporter(SpanExporter):
    """Collects finished spans in memory for test assertions."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


class TestBackgroundFieldInResponseObject:
    """Test that the background field is properly defined in OpenAIResponseObject."""

    def test_background_field_default_is_none(self):
        """Verify background field defaults to None."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="completed",
            output=[],
            store=True,
        )
        assert response.background is None

    def test_background_field_can_be_true(self):
        """Verify background field can be set to True."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="queued",
            output=[],
            background=True,
            store=True,
        )
        assert response.background is True

    def test_background_field_can_be_false(self):
        """Verify background field can be False."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="completed",
            output=[],
            background=False,
            store=True,
        )
        assert response.background is False


class TestResponseStatus:
    """Test that all expected status values work correctly."""

    @pytest.mark.parametrize(
        "status",
        ["queued", "in_progress", "completed", "failed", "incomplete"],
    )
    def test_valid_status_values(self, status):
        """Verify all OpenAI-compatible status values are accepted."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status=status,
            output=[],
            background=True if status in ("queued", "in_progress") else False,
            store=True,
        )
        assert response.status == status

    def test_queued_status_with_background(self):
        """Verify queued status is typically used with background=True."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="queued",
            output=[],
            background=True,
            store=True,
        )
        assert response.status == "queued"
        assert response.background is True


class TestResponseObjectSerialization:
    """Test that the response object serializes correctly with background field."""

    def test_model_dump_includes_background(self):
        """Verify model_dump includes the background field."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="queued",
            output=[],
            background=True,
            store=True,
        )
        data = response.model_dump()
        assert "background" in data
        assert data["background"] is True

    def test_model_dump_json_includes_background(self):
        """Verify JSON serialization includes the background field."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="completed",
            output=[],
            background=False,
            store=True,
        )
        json_str = response.model_dump_json()
        assert '"background":false' in json_str or '"background": false' in json_str


class TestResponseErrorForBackground:
    """Test error responses for background processing failures."""

    def test_error_response_with_background(self):
        """Verify error responses can include background field."""
        error = OpenAIResponseError(
            code="processing_error",
            message="Background processing failed",
        )
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="failed",
            output=[],
            background=True,
            error=error,
            store=True,
        )
        assert response.status == "failed"
        assert response.background is True
        assert response.error is not None
        assert response.error.code == "processing_error"


def _make_responses_impl():
    """Create an OpenAIResponsesImpl with all dependencies mocked."""
    return OpenAIResponsesImpl(
        inference_api=AsyncMock(),
        tool_groups_api=AsyncMock(),
        tool_runtime_api=AsyncMock(),
        responses_store=AsyncMock(),
        vector_io_api=AsyncMock(),
        safety_api=None,
        conversations_api=AsyncMock(),
        prompts_api=AsyncMock(),
        files_api=AsyncMock(),
        connectors_api=AsyncMock(),
    )


class TestResponsesOtelContextPropagation:
    """Verify that OTel trace context flows correctly through the background worker queue.

    The responses worker runs a full multi-step loop (_run_background_response_loop)
    containing status updates, LLM calls, tool execution, and DB writes. All of
    these operations must be attributed to the originating request's trace, not
    to whichever request first spawned the worker.
    """

    async def test_worker_attributes_work_to_correct_request_trace(self):
        """Each queued response is processed under its originating request's trace context."""
        exporter = _CollectingExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        impl = _make_responses_impl()

        async def mock_response_loop(**kwargs):
            with tracer.start_as_current_span(f"process-{kwargs['response_id']}"):
                await asyncio.sleep(0)

        with patch.object(impl, "_run_background_response_loop", side_effect=mock_response_loop):
            worker_task = create_task_with_detached_otel_context(impl._background_worker())

            with tracer.start_as_current_span("request-A"):
                impl._background_queue.put_nowait(dict(response_id="resp-A", _otel_context=capture_otel_context()))

            with tracer.start_as_current_span("request-B"):
                impl._background_queue.put_nowait(dict(response_id="resp-B", _otel_context=capture_otel_context()))

            await impl._background_queue.join()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        provider.force_flush()
        spans_by_name = {s.name: s for s in exporter.spans}

        request_a_trace = spans_by_name["request-A"].context.trace_id
        request_b_trace = spans_by_name["request-B"].context.trace_id
        process_a_trace = spans_by_name["process-resp-A"].context.trace_id
        process_b_trace = spans_by_name["process-resp-B"].context.trace_id

        assert request_a_trace != request_b_trace, "Requests should have distinct traces"

        assert process_a_trace == request_a_trace, "Response processing for resp-A should be in request-A's trace"
        assert process_b_trace == request_b_trace, "Response processing for resp-B should be in request-B's trace"

    async def test_worker_does_not_leak_context_between_items(self):
        """After processing one item, the worker returns to a clean context.

        This ensures that if item A's processing sets some OTel state, it
        doesn't bleed into item B's processing.
        """
        exporter = _CollectingExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        impl = _make_responses_impl()
        trace_ids_during_processing = {}

        async def mock_response_loop(**kwargs):
            rid = kwargs["response_id"]
            # The parent span has ended by this point, but the context still
            # carries its trace_id. Child spans inherit this trace_id.
            span_ctx = trace.get_current_span().get_span_context()
            trace_ids_during_processing[rid] = span_ctx.trace_id if span_ctx.trace_id != 0 else None
            with tracer.start_as_current_span(f"work-{rid}"):
                await asyncio.sleep(0)

        with patch.object(impl, "_run_background_response_loop", side_effect=mock_response_loop):
            worker_task = create_task_with_detached_otel_context(impl._background_worker())

            with tracer.start_as_current_span("req-1"):
                impl._background_queue.put_nowait(dict(response_id="r1", _otel_context=capture_otel_context()))

            with tracer.start_as_current_span("req-2"):
                impl._background_queue.put_nowait(dict(response_id="r2", _otel_context=capture_otel_context()))

            await impl._background_queue.join()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        provider.force_flush()
        spans_by_name = {s.name: s for s in exporter.spans}

        req1_trace = spans_by_name["req-1"].context.trace_id
        req2_trace = spans_by_name["req-2"].context.trace_id

        assert trace_ids_during_processing["r1"] is not None, "r1 should have a trace context"
        assert trace_ids_during_processing["r2"] is not None, "r2 should have a trace context"
        assert trace_ids_during_processing["r1"] == req1_trace
        assert trace_ids_during_processing["r2"] == req2_trace

    async def test_error_handling_runs_under_request_context(self):
        """When processing fails, the error handler's DB writes are also in the request's trace."""
        exporter = _CollectingExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        impl = _make_responses_impl()
        mock_response = OpenAIResponseObject(
            id="resp-err",
            created_at=1234567890,
            model="test-model",
            status="in_progress",
            output=[],
            store=True,
        )
        impl.responses_store.get_response_object = AsyncMock(return_value=mock_response)
        impl.responses_store.update_response_object = AsyncMock()

        error_update_trace_ids = []
        original_update = impl.responses_store.update_response_object

        async def tracking_update(obj):
            span_ctx = trace.get_current_span().get_span_context()
            if span_ctx.trace_id != 0:
                error_update_trace_ids.append(span_ctx.trace_id)
            return await original_update(obj)

        impl.responses_store.update_response_object = tracking_update

        async def failing_loop(**kwargs):
            raise RuntimeError("simulated failure")

        with patch.object(impl, "_run_background_response_loop", side_effect=failing_loop):
            worker_task = create_task_with_detached_otel_context(impl._background_worker())

            with tracer.start_as_current_span("failing-request"):
                request_trace = trace.get_current_span().get_span_context().trace_id
                impl._background_queue.put_nowait(dict(response_id="resp-err", _otel_context=capture_otel_context()))

            await impl._background_queue.join()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        assert len(error_update_trace_ids) > 0, "Error handler should have made DB updates"
        for tid in error_update_trace_ids:
            assert tid == request_trace, "Error handler DB writes should be in the failing request's trace"
