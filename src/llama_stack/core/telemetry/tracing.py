# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import contextvars
import logging  # allow-direct-logging
import queue
import secrets
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Self

from llama_stack.core.telemetry.telemetry import (
    ROOT_SPAN_MARKERS,
    Event,
    LogSeverity,
    Span,
    SpanEndPayload,
    SpanStartPayload,
    SpanStatus,
    StructuredLogEvent,
    Telemetry,
    UnstructuredLogEvent,
)
from llama_stack.core.telemetry.trace_protocol import serialize_value
from llama_stack.log import get_logger

logger = get_logger(__name__, category="core")

# Fallback logger that does NOT propagate to TelemetryHandler to avoid recursion
_fallback_logger = logging.getLogger("llama_stack.telemetry.background")
if not _fallback_logger.handlers:
    _fallback_logger.propagate = False
    _fallback_logger.setLevel(logging.ERROR)
    _fallback_handler = logging.StreamHandler(sys.stderr)
    _fallback_handler.setLevel(logging.ERROR)
    _fallback_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _fallback_logger.addHandler(_fallback_handler)


INVALID_SPAN_ID = 0x0000000000000000
INVALID_TRACE_ID = 0x00000000000000000000000000000000

# The logical root span may not be visible to this process if a parent context
# is passed in. The local root span is the first local span in a trace.
LOCAL_ROOT_SPAN_MARKER = "__local_root_span__"


def trace_id_to_str(trace_id: int) -> str:
    """Convenience trace ID formatting method
    Args:
        trace_id: Trace ID int

    Returns:
        The trace ID as 32-byte hexadecimal string
    """
    return format(trace_id, "032x")


def span_id_to_str(span_id: int) -> str:
    """Convenience span ID formatting method
    Args:
        span_id: Span ID int

    Returns:
        The span ID as 16-byte hexadecimal string
    """
    return format(span_id, "016x")


def generate_span_id() -> str:
    span_id = secrets.randbits(64)
    while span_id == INVALID_SPAN_ID:
        span_id = secrets.randbits(64)
    return span_id_to_str(span_id)


def generate_trace_id() -> str:
    trace_id = secrets.randbits(128)
    while trace_id == INVALID_TRACE_ID:
        trace_id = secrets.randbits(128)
    return trace_id_to_str(trace_id)


LOG_QUEUE_FULL_LOG_INTERVAL_SECONDS = 60.0


class BackgroundLogger:
    def __init__(self, api: Telemetry, capacity: int = 100000):
        self.api = api
        self.log_queue: queue.Queue[Any] = queue.Queue(maxsize=capacity)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._last_queue_full_log_time: float = 0.0
        self._dropped_since_last_notice: int = 0

    def log_event(self, event: Event) -> None:
        try:
            self.log_queue.put_nowait(event)
        except queue.Full:
            # Aggregate drops and emit at most once per interval via fallback logger
            self._dropped_since_last_notice += 1
            current_time = time.time()
            if current_time - self._last_queue_full_log_time >= LOG_QUEUE_FULL_LOG_INTERVAL_SECONDS:
                _fallback_logger.error(
                    "Log queue is full; dropped %d events since last notice",
                    self._dropped_since_last_notice,
                )
                self._last_queue_full_log_time = current_time
                self._dropped_since_last_notice = 0

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._process_logs())

    async def _process_logs(self):
        while True:
            try:
                event = self.log_queue.get()
                await self.api.log_event(event)
            except Exception:
                import traceback

                traceback.print_exc()
                print("Error processing log event")
            finally:
                self.log_queue.task_done()

    def __del__(self) -> None:
        self.log_queue.join()


BACKGROUND_LOGGER: BackgroundLogger | None = None


def enqueue_event(event: Event) -> None:
    """Enqueue a telemetry event to the background logger if available.

    This provides a non-blocking path for routers and other hot paths to
    submit telemetry without awaiting the Telemetry API, reducing contention
    with the main event loop.
    """
    global BACKGROUND_LOGGER
    if BACKGROUND_LOGGER is None:
        raise RuntimeError("Telemetry API not initialized")
    BACKGROUND_LOGGER.log_event(event)


class TraceContext:
    def __init__(self, logger: BackgroundLogger, trace_id: str):
        self.logger = logger
        self.trace_id = trace_id
        self.spans: list[Span] = []

    def push_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        current_span = self.get_current_span()
        span = Span(
            span_id=generate_span_id(),
            trace_id=self.trace_id,
            name=name,
            start_time=datetime.now(UTC),
            parent_span_id=current_span.span_id if current_span else None,
            attributes=attributes,
        )

        self.logger.log_event(
            StructuredLogEvent(
                trace_id=span.trace_id,
                span_id=span.span_id,
                timestamp=span.start_time,
                attributes=span.attributes,
                payload=SpanStartPayload(
                    name=span.name,
                    parent_span_id=span.parent_span_id,
                ),
            )
        )

        self.spans.append(span)
        return span

    def pop_span(self, status: SpanStatus = SpanStatus.OK) -> None:
        span = self.spans.pop()
        if span is not None:
            self.logger.log_event(
                StructuredLogEvent(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    timestamp=span.start_time,
                    attributes=span.attributes,
                    payload=SpanEndPayload(
                        status=status,
                    ),
                )
            )

    def get_current_span(self) -> Span | None:
        return self.spans[-1] if self.spans else None


CURRENT_TRACE_CONTEXT: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trace_context", default=None
)


def setup_logger(api: Telemetry, level: int = logging.INFO):
    global BACKGROUND_LOGGER

    if BACKGROUND_LOGGER is None:
        BACKGROUND_LOGGER = BackgroundLogger(api)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(TelemetryHandler())


async def start_trace(name: str, attributes: dict[str, Any] | None = None) -> TraceContext | None:
    global CURRENT_TRACE_CONTEXT, BACKGROUND_LOGGER

    if BACKGROUND_LOGGER is None:
        logger.debug("No Telemetry implementation set. Skipping trace initialization...")
        return None

    trace_id = generate_trace_id()
    context = TraceContext(BACKGROUND_LOGGER, trace_id)
    # Mark this span as the root for the trace for now. The processing of
    # traceparent context if supplied comes later and will result in the
    # ROOT_SPAN_MARKERS being removed. Also mark this is the 'local' root,
    # i.e. the root of the spans originating in this process as this is
    # needed to ensure that we insert this 'local' root span's id into
    # the trace record in sqlite store.
    attributes = dict.fromkeys(ROOT_SPAN_MARKERS, True) | {LOCAL_ROOT_SPAN_MARKER: True} | (attributes or {})
    context.push_span(name, attributes)

    CURRENT_TRACE_CONTEXT.set(context)
    return context


async def end_trace(status: SpanStatus = SpanStatus.OK):
    global CURRENT_TRACE_CONTEXT

    context = CURRENT_TRACE_CONTEXT.get()
    if context is None:
        logger.debug("No trace context to end")
        return

    context.pop_span(status)
    CURRENT_TRACE_CONTEXT.set(None)


def severity(levelname: str) -> LogSeverity:
    if levelname == "DEBUG":
        return LogSeverity.DEBUG
    elif levelname == "INFO":
        return LogSeverity.INFO
    elif levelname == "WARNING":
        return LogSeverity.WARN
    elif levelname == "ERROR":
        return LogSeverity.ERROR
    elif levelname == "CRITICAL":
        return LogSeverity.CRITICAL
    else:
        raise ValueError(f"Unknown log level: {levelname}")


# TODO: ideally, the actual emitting should be done inside a separate daemon
# process completely isolated from the server
class TelemetryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # horrendous hack to avoid logging from asyncio and getting into an infinite loop
        if record.module in ("asyncio", "selector_events"):
            return

        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT.get()
        if context is None:
            return

        span = context.get_current_span()
        if span is None:
            return

        enqueue_event(
            UnstructuredLogEvent(
                trace_id=span.trace_id,
                span_id=span.span_id,
                timestamp=datetime.now(UTC),
                message=self.format(record),
                severity=severity(record.levelname),
            )
        )

    def close(self) -> None:
        pass


class SpanContextManager:
    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.attributes = attributes
        self.span: Span | None = None

    def __enter__(self) -> Self:
        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT.get()
        if not context:
            logger.debug("No trace context to push span")
            return self

        self.span = context.push_span(self.name, self.attributes)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT.get()
        if not context:
            logger.debug("No trace context to pop span")
            return

        context.pop_span()

    def set_attribute(self, key: str, value: Any) -> None:
        if self.span:
            if self.span.attributes is None:
                self.span.attributes = {}
            self.span.attributes[key] = serialize_value(value)

    async def __aenter__(self) -> Self:
        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT.get()
        if not context:
            logger.debug("No trace context to push span")
            return self

        self.span = context.push_span(self.name, self.attributes)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        global CURRENT_TRACE_CONTEXT
        context = CURRENT_TRACE_CONTEXT.get()
        if not context:
            logger.debug("No trace context to pop span")
            return

        context.pop_span()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if asyncio.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)
            else:
                return sync_wrapper(*args, **kwargs)

        return wrapper


def span(name: str, attributes: dict[str, Any] | None = None) -> SpanContextManager:
    return SpanContextManager(name, attributes)


def get_current_span() -> Span | None:
    global CURRENT_TRACE_CONTEXT
    if CURRENT_TRACE_CONTEXT is None:
        logger.debug("No trace context to get current span")
        return None

    context = CURRENT_TRACE_CONTEXT.get()
    if context:
        return context.get_current_span()
    return None
