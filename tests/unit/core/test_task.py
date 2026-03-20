# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from llama_stack.core.task import (
    activate_otel_context,
    capture_otel_context,
    create_task_with_detached_otel_context,
)


class _CollectingExporter(SpanExporter):
    """Collects finished spans in memory for test assertions."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


async def test_detached_task_runs_coroutine():
    """The helper creates a task that actually runs the coroutine to completion."""
    result = []

    async def work():
        result.append("done")

    task = create_task_with_detached_otel_context(work())
    await task
    assert result == ["done"]


async def test_detached_task_clears_otel_context():
    """The task should run with an empty OTel context, not the parent's."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    captured_span = {}

    async def capture_context():
        captured_span["inner"] = trace.get_current_span()

    with tracer.start_as_current_span("parent-span"):
        parent_ctx = otel_context.get_current()
        parent_span = trace.get_current_span()

        task = create_task_with_detached_otel_context(capture_context())
        await task

        assert not captured_span["inner"].is_recording()
        assert parent_span.is_recording()
        assert otel_context.get_current() == parent_ctx


async def test_detached_task_restores_caller_context():
    """The calling coroutine's OTel context is not affected by creating a detached task."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("parent-span"):
        before = otel_context.get_current()
        create_task_with_detached_otel_context(asyncio.sleep(0))
        after = otel_context.get_current()
        assert before == after


async def test_detached_task_produces_independent_trace():
    """Spans created inside a detached task belong to a separate trace, not the parent's."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    async def background_work():
        with tracer.start_as_current_span("background-db-write"):
            await asyncio.sleep(0)

    with tracer.start_as_current_span("http-request"):
        task = create_task_with_detached_otel_context(background_work())
        await task

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    request_span = span_by_name["http-request"]
    bg_span = span_by_name["background-db-write"]

    assert request_span.context.trace_id != bg_span.context.trace_id, (
        "Background span should belong to a different trace than the request"
    )
    assert bg_span.parent is None, "Background span should be a root span with no parent"


async def test_normal_child_task_shares_trace():
    """Contrast: a regular asyncio.create_task DOES inherit the parent trace."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    async def child_work():
        with tracer.start_as_current_span("child-span"):
            await asyncio.sleep(0)

    with tracer.start_as_current_span("parent-request"):
        task = asyncio.create_task(child_work())
        await task

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    parent_span = span_by_name["parent-request"]
    child_span = span_by_name["child-span"]

    assert parent_span.context.trace_id == child_span.context.trace_id, (
        "Regular create_task should share the parent's trace"
    )


async def test_capture_and_attach_otel_context():
    """capture_otel_context snapshots the current context; activate_otel_context re-activates it."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("request"):
        ctx = capture_otel_context()
        request_trace_id = trace.get_current_span().get_span_context().trace_id

    with activate_otel_context(ctx):
        with tracer.start_as_current_span("reattached-work"):
            reattached_trace_id = trace.get_current_span().get_span_context().trace_id

    assert request_trace_id == reattached_trace_id, "Work done under attached context should share the original trace"


async def test_attached_context_restores_on_exit():
    """activate_otel_context restores the previous context when the block exits."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("outer"):
        outer_ctx = otel_context.get_current()

        inner_ctx = otel_context.Context()
        with activate_otel_context(inner_ctx):
            assert otel_context.get_current() == inner_ctx

        assert otel_context.get_current() == outer_ctx


async def test_context_through_queue_pattern():
    """End-to-end: context captured at enqueue time is correctly attached in a detached worker.

    This simulates the inference_store pattern:
    1. Request creates a span and enqueues work with captured context
    2. Worker runs in a detached (empty) context
    3. Worker attaches the captured context before processing
    4. The resulting span belongs to the original request's trace
    """
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    queue: asyncio.Queue[tuple[str, otel_context.Context]] = asyncio.Queue()

    async def worker():
        item, ctx = await queue.get()
        with activate_otel_context(ctx):
            with tracer.start_as_current_span(f"db-write-{item}"):
                await asyncio.sleep(0)
        queue.task_done()

    with tracer.start_as_current_span("http-request-A"):
        ctx_a = capture_otel_context()
        await queue.put(("A", ctx_a))

    worker_task = create_task_with_detached_otel_context(worker())
    await worker_task
    await queue.join()

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    request_span = span_by_name["http-request-A"]
    write_span = span_by_name["db-write-A"]

    assert request_span.context.trace_id == write_span.context.trace_id, (
        "DB write should belong to the same trace as the originating request"
    )


async def test_context_through_queue_no_cross_contamination():
    """Two requests enqueue work; each DB write is attributed to its own request trace.

    This is the key property: workers don't permanently inherit any single
    request's context, and each queued item carries the correct context.
    """
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    queue: asyncio.Queue[tuple[str, otel_context.Context]] = asyncio.Queue()
    processed = asyncio.Event()

    async def worker():
        for _ in range(2):
            item, ctx = await queue.get()
            with activate_otel_context(ctx):
                with tracer.start_as_current_span(f"db-write-{item}"):
                    await asyncio.sleep(0)
            queue.task_done()
        processed.set()

    worker_task = create_task_with_detached_otel_context(worker())

    with tracer.start_as_current_span("request-A"):
        await queue.put(("A", capture_otel_context()))

    with tracer.start_as_current_span("request-B"):
        await queue.put(("B", capture_otel_context()))

    await processed.wait()
    await worker_task

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    request_a = span_by_name["request-A"]
    request_b = span_by_name["request-B"]
    write_a = span_by_name["db-write-A"]
    write_b = span_by_name["db-write-B"]

    assert write_a.context.trace_id == request_a.context.trace_id, "Write A should be in request A's trace"
    assert write_b.context.trace_id == request_b.context.trace_id, "Write B should be in request B's trace"
    assert request_a.context.trace_id != request_b.context.trace_id, "Request A and B should have different traces"
