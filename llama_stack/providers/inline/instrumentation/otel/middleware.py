# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

from opentelemetry import trace
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="instrumentation::otel")


class StreamingMetricsMiddleware:
    """
    ASGI middleware to track streaming response metrics.

    :param app: The ASGI app to wrap
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        logger.debug(f"StreamingMetricsMiddleware called for {scope.get('method')} {scope.get('path')}")
        start_time = time.time()
        is_streaming = False

        async def send_wrapper(message: Message):
            nonlocal is_streaming

            # Detect streaming responses by headers
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                for name, value in headers:
                    if name == b"content-type" and b"text/event-stream" in value:
                        is_streaming = True
                        # Add streaming attribute to current span
                        current_span = trace.get_current_span()
                        if current_span and current_span.is_recording():
                            current_span.set_attribute("http.response.is_streaming", True)
                        break

            # Record total duration when response body completes
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                if is_streaming:
                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        total_duration_ms = (time.time() - start_time) * 1000
                        current_span.set_attribute("http.streaming.total_duration_ms", total_duration_ms)

            await send(message)

        await self.app(scope, receive, send_wrapper)


class MetricsSpanExporter(SpanExporter):
    """
    Records additional custom HTTP metrics during otel span export.

    :param request_duration: Histogram to record request duration
    :param streaming_duration: Histogram to record streaming duration
    :param streaming_requests: Counter to record streaming requests
    :param request_count: Counter to record request count
    """

    def __init__(
        self,
        request_duration: Histogram,
        streaming_duration: Histogram,
        streaming_requests: Counter,
        request_count: Counter,
    ):
        self.request_duration = request_duration
        self.streaming_duration = streaming_duration
        self.streaming_requests = streaming_requests
        self.request_count = request_count

    def export(self, spans):
        for span in spans:
            if not span.attributes or not span.attributes.get("http.method"):
                continue
            logger.debug(f"Processing span: {span.name}")

            if span.end_time is None or span.start_time is None:
                continue

            duration_ms = (span.end_time - span.start_time) / 1_000_000
            is_streaming = span.attributes.get("http.response.is_streaming", False)

            attributes = {
                "http.method": str(span.attributes.get("http.method", "UNKNOWN")),
                "http.route": str(span.attributes.get("http.route", span.attributes.get("http.target", "/"))),
                "http.status_code": str(span.attributes.get("http.status_code", 0)),
                "trace_id": str(span.attributes.get("trace_id", "")),
                "span_id": str(span.attributes.get("span_id", "")),
            }

            # Record request count and duration
            logger.debug(f"Recording metrics: duration={duration_ms}ms, attributes={attributes}")
            self.request_count.add(1, attributes)
            self.request_duration.record(duration_ms, attributes)

            if is_streaming:
                logger.debug(f"MetricsSpanExporter: Recording streaming metrics for {span.name}")
                self.streaming_requests.add(1, attributes)
                stream_duration = span.attributes.get("http.streaming.total_duration_ms")
                if stream_duration and isinstance(stream_duration, (int | float)):
                    self.streaming_duration.record(float(stream_duration), attributes)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass
