import os
import threading

from opentelemetry import trace, metrics
from opentelemetry.context.context import Context
from opentelemetry.sdk.resources import Attributes, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.metrics import Counter, UpDownCounter, Histogram, ObservableGauge
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Span, SpanKind, _Links
from typing import Sequence
from pydantic import PrivateAttr

from llama_stack.core.telemetry.tracing import TelemetryProvider
from llama_stack.log import get_logger

from .config import OTelTelemetryConfig
from fastapi import FastAPI


logger = get_logger(name=__name__, category="telemetry::otel")


class OTelTelemetryProvider(TelemetryProvider):
    """
    A simple Open Telemetry native telemetry provider.
    """
    config: OTelTelemetryConfig
    _counters: dict[str, Counter] = PrivateAttr(default_factory=dict)
    _up_down_counters: dict[str, UpDownCounter] = PrivateAttr(default_factory=dict)
    _histograms: dict[str, Histogram] = PrivateAttr(default_factory=dict)
    _gauges: dict[str, ObservableGauge] = PrivateAttr(default_factory=dict)


    def model_post_init(self, __context):
        """Initialize provider after Pydantic validation."""
        self._lock = threading.Lock()

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
                logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is not set. Traces will not be exported.")
            if not os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
                logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_METRICS_ENDPOINT is not set. Metrics will not be exported.")

    def fastapi_middleware(self, app: FastAPI):
        FastAPIInstrumentor.instrument_app(app)

    def custom_trace(self, 
    name: str,
    context: Context | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Attributes = {},
    links: _Links = None,
    start_time: int | None = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True) -> Span:
        """
        Creates a custom tracing span using the Open Telemetry SDK.
        """
        tracer = trace.get_tracer(__name__)
        return tracer.start_span(name, context, kind, attributes, links, start_time, record_exception, set_status_on_exception)


    def record_count(self, name: str, amount: int|float, context: Context | None = None, attributes: dict[str, str] | None = None, unit: str = "", description: str = ""):
        """
        Increments a counter metric using the Open Telemetry SDK that are indexed by the meter name.
        This function is designed to be compatible with other popular telemetry providers design patterns,
        like Datadog and New Relic.
        """
        meter = metrics.get_meter(__name__)

        with self._lock:
            if name not in self._counters:
                self._counters[name] = meter.create_counter(name, unit=unit, description=description)
            counter = self._counters[name]

        counter.add(amount, attributes=attributes, context=context)


    def record_histogram(self, name: str, value: int|float, context: Context | None = None, attributes: dict[str, str] | None = None, unit: str = "", description: str = "", explicit_bucket_boundaries_advisory: Sequence[float] | None = None):
        """
        Records a histogram metric using the Open Telemetry SDK that are indexed by the meter name.
        This function is designed to be compatible with other popular telemetry providers design patterns,
        like Datadog and New Relic.
        """
        meter = metrics.get_meter(__name__)

        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = meter.create_histogram(name, unit=unit, description=description, explicit_bucket_boundaries_advisory=explicit_bucket_boundaries_advisory)
            histogram = self._histograms[name]

        histogram.record(value, attributes=attributes, context=context)


    def record_up_down_counter(self, name: str, value: int|float, context: Context | None = None, attributes: dict[str, str] | None = None, unit: str = "", description: str = ""):
        """
        Records an up/down counter metric using the Open Telemetry SDK that are indexed by the meter name.
        This function is designed to be compatible with other popular telemetry providers design patterns,
        like Datadog and New Relic.
        """
        meter = metrics.get_meter(__name__)

        with self._lock:
            if name not in self._up_down_counters:
                self._up_down_counters[name] = meter.create_up_down_counter(name, unit=unit, description=description)
            up_down_counter = self._up_down_counters[name]

        up_down_counter.add(value, attributes=attributes, context=context)
