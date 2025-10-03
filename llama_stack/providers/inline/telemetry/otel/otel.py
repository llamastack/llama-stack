# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Attributes, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from sqlalchemy import Engine

from llama_stack.core.telemetry.telemetry import TelemetryProvider
from llama_stack.log import get_logger

from .config import OTelTelemetryConfig

logger = get_logger(name=__name__, category="telemetry::otel")


class OTelTelemetryProvider(TelemetryProvider):
    """
    A simple Open Telemetry native telemetry provider.
    """

    config: OTelTelemetryConfig

    def model_post_init(self, __context):
        """Initialize provider after Pydantic validation."""

        attributes: Attributes = {
            key: value
            for key, value in {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.deployment_environment,
            }.items()
            if value is not None
        }

        resource = Resource.create(attributes)

        # Configure the tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        otlp_span_exporter = OTLPSpanExporter()

        # Configure the span processor
        # Enable batching of spans to reduce the number of requests to the collector
        if self.config.span_processor == "batch":
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))
        elif self.config.span_processor == "simple":
            tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_span_exporter))

        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)

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

    def fastapi_middleware(self, app: FastAPI):
        """
        Instrument FastAPI with OTel for automatic tracing and metrics.

        Captures:
        - Distributed traces for all HTTP requests (via FastAPIInstrumentor)
        - HTTP metrics following semantic conventions (custom middleware)
        """
        # Enable automatic tracing
        FastAPIInstrumentor.instrument_app(app)

        # Add custom middleware for HTTP metrics
        meter = metrics.get_meter("llama_stack.http.server")

        # Create HTTP metrics following semantic conventions
        # https://opentelemetry.io/docs/specs/semconv/http/http-metrics/
        request_duration = meter.create_histogram(
            "http.server.request.duration", unit="ms", description="Duration of HTTP server requests"
        )

        active_requests = meter.create_up_down_counter(
            "http.server.active_requests", unit="requests", description="Number of active HTTP server requests"
        )

        request_count = meter.create_counter(
            "http.server.request.count", unit="requests", description="Total number of HTTP server requests"
        )

        # Add middleware to record metrics
        @app.middleware("http")  # type: ignore[misc]
        async def http_metrics_middleware(request, call_next):
            import time

            # Record active request
            active_requests.add(
                1,
                {
                    "http.method": request.method,
                    "http.route": request.url.path,
                },
            )

            start_time = time.time()
            status_code = 500  # Default to error

            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception:
                raise
            finally:
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000

                attributes = {
                    "http.method": request.method,
                    "http.route": request.url.path,
                    "http.status_code": status_code,
                }

                request_duration.record(duration_ms, attributes)
                request_count.add(1, attributes)
                active_requests.add(
                    -1,
                    {
                        "http.method": request.method,
                        "http.route": request.url.path,
                    },
                )

            return response

    def sqlalchemy_instrumentation(self, engine: Engine | None = None):
        kwargs = {}
        if engine:
            kwargs["engine"] = engine
        SQLAlchemyInstrumentor().instrument(**kwargs)
