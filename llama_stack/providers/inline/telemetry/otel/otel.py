# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import time

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from sqlalchemy import Engine
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llama_stack.core.telemetry.telemetry import TelemetryProvider
from llama_stack.log import get_logger

from .config import OTelTelemetryConfig

logger = get_logger(name=__name__, category="telemetry::otel")


class StreamingMetricsMiddleware:
    """
    Pure ASGI middleware to track streaming response metrics.

    This follows Starlette best practices by implementing pure ASGI,
    which is more efficient and less prone to bugs than BaseHTTPMiddleware.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        logger.debug(f"StreamingMetricsMiddleware called for {scope.get('method')} {scope.get('path')}")
        start_time = time.time()

        # Track if this is a streaming response
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
    """Records HTTP metrics from span data."""

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
        logger.debug(f"MetricsSpanExporter.export called with {len(spans)} spans")
        for span in spans:
            if not span.attributes or not span.attributes.get("http.method"):
                continue
            logger.debug(f"Processing span: {span.name}")

            if span.end_time is None or span.start_time is None:
                continue

            # Calculate time-to-first-byte duration
            duration_ns = span.end_time - span.start_time
            duration_ms = duration_ns / 1_000_000

            # Check if this was a streaming response
            is_streaming = span.attributes.get("http.response.is_streaming", False)

            attributes = {
                "http.method": str(span.attributes.get("http.method", "UNKNOWN")),
                "http.route": str(span.attributes.get("http.route", span.attributes.get("http.target", "/"))),
                "http.status_code": str(span.attributes.get("http.status_code", 0)),
            }

            # set distributed trace attributes
            if span.attributes.get("trace_id"):
                attributes["trace_id"] = str(span.attributes.get("trace_id"))
            if span.attributes.get("span_id"):
                attributes["span_id"] = str(span.attributes.get("span_id"))

            # Record request count and duration
            logger.debug(f"Recording metrics: duration={duration_ms}ms, attributes={attributes}")
            self.request_count.add(1, attributes)
            self.request_duration.record(duration_ms, attributes)
            logger.debug("Metrics recorded successfully")

            # For streaming, record separately
            if is_streaming:
                logger.debug(f"MetricsSpanExporter: Recording streaming metrics for {span.name}")
                self.streaming_requests.add(1, attributes)

                # If full streaming duration is available
                stream_total_duration = span.attributes.get("http.streaming.total_duration_ms")
                if stream_total_duration and isinstance(stream_total_duration, int | float):
                    logger.debug(f"MetricsSpanExporter: Recording streaming duration: {stream_total_duration}ms")
                    self.streaming_duration.record(float(stream_total_duration), attributes)
                else:
                    logger.warning(
                        "MetricsSpanExporter: Streaming span has no http.streaming.total_duration_ms attribute"
                    )

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


# NOTE: DO NOT ALLOW LLM TO MODIFY THIS WITHOUT TESTING AND SUPERVISION: it frequently breaks otel integrations
class OTelTelemetryProvider(TelemetryProvider):
    """
    A simple Open Telemetry native telemetry provider.
    """

    config: OTelTelemetryConfig

    def model_post_init(self, __context):
        """Initialize provider after Pydantic validation."""

        # Do not fail the application, but warn the user if the endpoints are not set properly.
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            if not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
                logger.warning(
                    "OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is not set. Traces will not be exported."
                )
            if not os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
                logger.warning(
                    "OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_METRICS_ENDPOINT is not set. Metrics will not be exported."
                )

        # Respect OTEL design standards where environment variables get highest precedence
        service_name = os.environ.get("OTEL_SERVICE_NAME")
        if not service_name:
            service_name = self.config.service_name

        # Create resource with service name
        resource = Resource.create({"service.name": service_name})

        # Configure the tracer provider (always, since llama stack run spawns subprocess without opentelemetry-instrument)
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Configure OTLP span exporter
        otlp_span_exporter = OTLPSpanExporter()

        # Add span processor (simple for immediate export, batch for performance)
        span_processor_type = os.environ.get("OTEL_SPAN_PROCESSOR", "batch")
        if span_processor_type == "batch":
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))
        else:
            tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_span_exporter))

        # Configure meter provider with OTLP exporter for metrics
        metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        logger.info(
            f"Initialized OpenTelemetry provider with service.name={service_name}, span_processor={span_processor_type}"
        )

    def fastapi_middleware(self, app: FastAPI):
        """
        Instrument FastAPI with OTel for automatic tracing and metrics.

        Captures telemetry for both regular and streaming HTTP requests:
        - Distributed traces (via FastAPIInstrumentor)
        - HTTP request metrics (count, duration, status)
        - Streaming-specific metrics (time-to-first-byte, total stream duration)
        """

        # Create meter for HTTP metrics
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
        # which will be called on first request (after routes are added), so hooks should work.
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
        logger.debug(f"TracerProvider: {provider}")
        if isinstance(provider, TracerProvider):
            metrics_exporter = MetricsSpanExporter(
                request_duration=request_duration,
                streaming_duration=streaming_duration,
                streaming_requests=streaming_requests,
                request_count=request_count,
            )
            provider.add_span_processor(BatchSpanProcessor(metrics_exporter))
            logger.debug("Added MetricsSpanExporter as BatchSpanProcessor")
        else:
            logger.warning(
                f"TracerProvider is not TracerProvider instance, it's {type(provider)}. MetricsSpanExporter not added."
            )
