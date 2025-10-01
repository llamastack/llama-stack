import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Attributes, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from llama_stack.core.telemetry.tracing import TelemetryProvider
from llama_stack.log import get_logger

from .config import OTelTelemetryConfig
from fastapi import FastAPI


logger = get_logger(name=__name__, category="telemetry::otel")


class OTelTelemetryProvider(TelemetryProvider):
    """
    A simple Open Telemetry native telemetry provider.
    """
    def __init__(self, config: OTelTelemetryConfig):
        self.config = config
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

        # Do not fail the application, but warn the user if the endpoints are not set properly
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            if not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
                logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is not set. Traces will not be exported.")
            if not os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
                logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_METRICS_ENDPOINT is not set. Metrics will not be exported.")

    def fastapi_middleware(self, app: FastAPI):
        FastAPIInstrumentor.instrument_app(app)
