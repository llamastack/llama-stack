# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os

from llama_stack.log import get_logger
from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from llama_stack.telemetry.middleware import StreamingMetricsMiddleware, MetricsSpanExporter


logger = get_logger(name=__name__, category="telemetry::otel")


class OTelInstrumentation:
    """OpenTelemetry instrumentation."""

    def fastapi_middleware(self, app: FastAPI):
        """Inject OpenTelemetry middleware into FastAPI."""
        meter = metrics.get_meter("llama_stack.http.server")

        # HTTP Metrics following OTel semantic conventions
        # https://opentelemetry.io/docs/specs/semconv/http/http-metrics/
        request_duration = meter.create_histogram(
            "http.server.request.duration",
            unit="ms",
            description="Duration of HTTP requests (time-to-first-byte for streaming)",
        )

        streaming_duration = meter.create_histogram(
            "http.server.streaming.duration",
            unit="ms",
            description="Total duration of streaming responses (from start to stream completion)",
        )

        request_count = meter.create_counter(
            "http.server.request.count", unit="requests", description="Total number of HTTP requests"
        )

        streaming_requests = meter.create_counter(
            "http.server.streaming.count", unit="requests", description="Number of streaming requests"
        )

        # Hook to enrich spans and record initial metrics
        def server_request_hook(span, scope):
            """
            Called by FastAPIInstrumentor for each request.

            This only reads from scope (ASGI dict), never touches request body.
            Safe to use without interfering with body parsing.
            """
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "/")

            # Add custom attributes
            span.set_attribute("service.component", "llama-stack-api")
            span.set_attribute("http.request", path)
            span.set_attribute("http.method", method)

            attributes = {
                "http.request": path,
                "http.method": method,
                "trace_id": span.attributes.get("trace_id", ""),
                "span_id": span.attributes.get("span_id", ""),
            }

            request_count.add(1, attributes)
            logger.debug(f"server_request_hook: recorded request_count for {method} {path}, attributes={attributes}")

        # NOTE: This is called BEFORE routes are added to the app
        # FastAPIInstrumentor.instrument_app() patches build_middleware_stack(),
        # which will be called on first request (after routes are added)
        logger.debug("Instrumenting FastAPI (routes will be added later)")
        FastAPIInstrumentor.instrument_app(
            app,
            server_request_hook=server_request_hook,
        )
        logger.debug(f"FastAPI instrumented: {getattr(app, '_is_instrumented_by_opentelemetry', False)}")

        # Add pure ASGI middleware for streaming metrics (always add, regardless of instrumentation)
        app.add_middleware(StreamingMetricsMiddleware)

        # Add metrics span processor
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            metrics_exporter = MetricsSpanExporter(
                request_duration=request_duration,
                streaming_duration=streaming_duration,
                streaming_requests=streaming_requests,
                request_count=request_count,
            )
            provider.add_span_processor(BatchSpanProcessor(metrics_exporter))
