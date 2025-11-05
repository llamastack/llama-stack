# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    Literal,
    cast,
)

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, Field

from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import Primitive
from llama_stack.schema_utils import json_schema_type, register_schema

ROOT_SPAN_MARKERS = ["__root__", "__root_span__"]

# Type alias for OpenTelemetry attribute values (excludes None)
AttributeValue = str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
Attributes = Mapping[str, AttributeValue]


@json_schema_type
class SpanStatus(Enum):
    """The status of a span indicating whether it completed successfully or with an error.
    :cvar OK: Span completed successfully without errors
    :cvar ERROR: Span completed with an error or failure
    """

    OK = "ok"
    ERROR = "error"


@json_schema_type
class Span(BaseModel):
    """A span representing a single operation within a trace.
    :param span_id: Unique identifier for the span
    :param trace_id: Unique identifier for the trace this span belongs to
    :param parent_span_id: (Optional) Unique identifier for the parent span, if this is a child span
    :param name: Human-readable name describing the operation this span represents
    :param start_time: Timestamp when the operation began
    :param end_time: (Optional) Timestamp when the operation finished, if completed
    :param attributes: (Optional) Key-value pairs containing additional metadata about the span
    """

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str
    start_time: datetime
    end_time: datetime | None = None
    attributes: dict[str, Any] | None = Field(default_factory=lambda: {})

    def set_attribute(self, key: str, value: Any):
        if self.attributes is None:
            self.attributes = {}
        self.attributes[key] = value


@json_schema_type
class Trace(BaseModel):
    """A trace representing the complete execution path of a request across multiple operations.
    :param trace_id: Unique identifier for the trace
    :param root_span_id: Unique identifier for the root span that started this trace
    :param start_time: Timestamp when the trace began
    :param end_time: (Optional) Timestamp when the trace finished, if completed
    """

    trace_id: str
    root_span_id: str
    start_time: datetime
    end_time: datetime | None = None


@json_schema_type
class EventType(Enum):
    """The type of telemetry event being logged.
    :cvar UNSTRUCTURED_LOG: A simple log message with severity level
    :cvar STRUCTURED_LOG: A structured log event with typed payload data
    :cvar METRIC: A metric measurement with value and unit
    """

    UNSTRUCTURED_LOG = "unstructured_log"
    STRUCTURED_LOG = "structured_log"
    METRIC = "metric"


@json_schema_type
class LogSeverity(Enum):
    """The severity level of a log message.
    :cvar VERBOSE: Detailed diagnostic information for troubleshooting
    :cvar DEBUG: Debug information useful during development
    :cvar INFO: General informational messages about normal operation
    :cvar WARN: Warning messages about potentially problematic situations
    :cvar ERROR: Error messages indicating failures that don't stop execution
    :cvar CRITICAL: Critical error messages indicating severe failures
    """

    VERBOSE = "verbose"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class EventCommon(BaseModel):
    """Common fields shared by all telemetry events.
    :param trace_id: Unique identifier for the trace this event belongs to
    :param span_id: Unique identifier for the span this event belongs to
    :param timestamp: Timestamp when the event occurred
    :param attributes: (Optional) Key-value pairs containing additional metadata about the event
    """

    trace_id: str
    span_id: str
    timestamp: datetime
    attributes: dict[str, Primitive] | None = Field(default_factory=lambda: {})


@json_schema_type
class UnstructuredLogEvent(EventCommon):
    """An unstructured log event containing a simple text message.
    :param type: Event type identifier set to UNSTRUCTURED_LOG
    :param message: The log message text
    :param severity: The severity level of the log message
    """

    type: Literal[EventType.UNSTRUCTURED_LOG] = EventType.UNSTRUCTURED_LOG
    message: str
    severity: LogSeverity


@json_schema_type
class MetricEvent(EventCommon):
    """A metric event containing a measured value.
    :param type: Event type identifier set to METRIC
    :param metric: The name of the metric being measured
    :param value: The numeric value of the metric measurement
    :param unit: The unit of measurement for the metric value
    """

    type: Literal[EventType.METRIC] = EventType.METRIC
    metric: str  # this would be an enum
    value: int | float
    unit: str


@json_schema_type
class StructuredLogType(Enum):
    """The type of structured log event payload.
    :cvar SPAN_START: Event indicating the start of a new span
    :cvar SPAN_END: Event indicating the completion of a span
    """

    SPAN_START = "span_start"
    SPAN_END = "span_end"


@json_schema_type
class SpanStartPayload(BaseModel):
    """Payload for a span start event.
    :param type: Payload type identifier set to SPAN_START
    :param name: Human-readable name describing the operation this span represents
    :param parent_span_id: (Optional) Unique identifier for the parent span, if this is a child span
    """

    type: Literal[StructuredLogType.SPAN_START] = StructuredLogType.SPAN_START
    name: str
    parent_span_id: str | None = None


@json_schema_type
class SpanEndPayload(BaseModel):
    """Payload for a span end event.
    :param type: Payload type identifier set to SPAN_END
    :param status: The final status of the span indicating success or failure
    """

    type: Literal[StructuredLogType.SPAN_END] = StructuredLogType.SPAN_END
    status: SpanStatus


StructuredLogPayload = Annotated[
    SpanStartPayload | SpanEndPayload,
    Field(discriminator="type"),
]
register_schema(StructuredLogPayload, name="StructuredLogPayload")


@json_schema_type
class StructuredLogEvent(EventCommon):
    """A structured log event containing typed payload data.
    :param type: Event type identifier set to STRUCTURED_LOG
    :param payload: The structured payload data for the log event
    """

    type: Literal[EventType.STRUCTURED_LOG] = EventType.STRUCTURED_LOG
    payload: StructuredLogPayload


Event = Annotated[
    UnstructuredLogEvent | MetricEvent | StructuredLogEvent,
    Field(discriminator="type"),
]
register_schema(Event, name="Event")


@json_schema_type
class EvalTrace(BaseModel):
    """A trace record for evaluation purposes.
    :param session_id: Unique identifier for the evaluation session
    :param step: The evaluation step or phase identifier
    :param input: The input data for the evaluation
    :param output: The actual output produced during evaluation
    :param expected_output: The expected output for comparison during evaluation
    """

    session_id: str
    step: str
    input: str
    output: str
    expected_output: str


@json_schema_type
class SpanWithStatus(Span):
    """A span that includes status information.
    :param status: (Optional) The current status of the span
    """

    status: SpanStatus | None = None


@json_schema_type
class QueryConditionOp(Enum):
    """Comparison operators for query conditions.
    :cvar EQ: Equal to comparison
    :cvar NE: Not equal to comparison
    :cvar GT: Greater than comparison
    :cvar LT: Less than comparison
    """

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"


@json_schema_type
class QueryCondition(BaseModel):
    """A condition for filtering query results.
    :param key: The attribute key to filter on
    :param op: The comparison operator to apply
    :param value: The value to compare against
    """

    key: str
    op: QueryConditionOp
    value: Any


class QueryTracesResponse(BaseModel):
    """Response containing a list of traces.
    :param data: List of traces matching the query criteria
    """

    data: list[Trace]


class QuerySpansResponse(BaseModel):
    """Response containing a list of spans.
    :param data: List of spans matching the query criteria
    """

    data: list[Span]


class QuerySpanTreeResponse(BaseModel):
    """Response containing a tree structure of spans.
    :param data: Dictionary mapping span IDs to spans with status information
    """

    data: dict[str, SpanWithStatus]


class MetricQueryType(Enum):
    """The type of metric query to perform.
    :cvar RANGE: Query metrics over a time range
    :cvar INSTANT: Query metrics at a specific point in time
    """

    RANGE = "range"
    INSTANT = "instant"


class MetricLabelOperator(Enum):
    """Operators for matching metric labels.
    :cvar EQUALS: Label value must equal the specified value
    :cvar NOT_EQUALS: Label value must not equal the specified value
    :cvar REGEX_MATCH: Label value must match the specified regular expression
    :cvar REGEX_NOT_MATCH: Label value must not match the specified regular expression
    """

    EQUALS = "="
    NOT_EQUALS = "!="
    REGEX_MATCH = "=~"
    REGEX_NOT_MATCH = "!~"


class MetricLabelMatcher(BaseModel):
    """A matcher for filtering metrics by label values.
    :param name: The name of the label to match
    :param value: The value to match against
    :param operator: The comparison operator to use for matching
    """

    name: str
    value: str
    operator: MetricLabelOperator = MetricLabelOperator.EQUALS


@json_schema_type
class MetricLabel(BaseModel):
    """A label associated with a metric.
    :param name: The name of the label
    :param value: The value of the label
    """

    name: str
    value: str


@json_schema_type
class MetricDataPoint(BaseModel):
    """A single data point in a metric time series.
    :param timestamp: Unix timestamp when the metric value was recorded
    :param value: The numeric value of the metric at this timestamp
    """

    timestamp: int
    value: float
    unit: str


@json_schema_type
class MetricSeries(BaseModel):
    """A time series of metric data points.
    :param metric: The name of the metric
    :param labels: List of labels associated with this metric series
    :param values: List of data points in chronological order
    """

    metric: str
    labels: list[MetricLabel]
    values: list[MetricDataPoint]


class QueryMetricsResponse(BaseModel):
    """Response containing metric time series data.
    :param data: List of metric series matching the query criteria
    """

    data: list[MetricSeries]


_GLOBAL_STORAGE: dict[str, dict[str | int, Any]] = {
    "active_spans": {},
    "counters": {},
    "gauges": {},
    "up_down_counters": {},
    "histograms": {},
}
_global_lock = threading.Lock()
_TRACER_PROVIDER = None

logger = get_logger(name=__name__, category="telemetry")


def _clean_attributes(attrs: dict[str, Any] | None) -> Attributes | None:
    """Remove None values from attributes dict to match OpenTelemetry's expected type."""
    if attrs is None:
        return None
    return {k: v for k, v in attrs.items() if v is not None}


def is_tracing_enabled(tracer):
    with tracer.start_as_current_span("check_tracing") as span:
        return span.is_recording()


class Telemetry:
    def __init__(self) -> None:
        self.meter = None

        global _TRACER_PROVIDER
        # Initialize the correct span processor based on the provider state.
        # This is needed since once the span processor is set, it cannot be unset.
        # Recreating the telemetry adapter multiple times will result in duplicate span processors.
        # Since the library client can be recreated multiple times in a notebook,
        # the kernel will hold on to the span processor and cause duplicate spans to be written.
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            if _TRACER_PROVIDER is None:
                provider = TracerProvider()
                trace.set_tracer_provider(provider)
                _TRACER_PROVIDER = provider

                # Use single OTLP endpoint for all telemetry signals

                # Let OpenTelemetry SDK handle endpoint construction automatically
                # The SDK will read OTEL_EXPORTER_OTLP_ENDPOINT and construct appropriate URLs
                # https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter
                span_exporter = OTLPSpanExporter()
                span_processor = BatchSpanProcessor(span_exporter)
                cast(TracerProvider, trace.get_tracer_provider()).add_span_processor(span_processor)

                metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
                metric_provider = MeterProvider(metric_readers=[metric_reader])
                metrics.set_meter_provider(metric_provider)
            self.is_otel_endpoint_set = True
        else:
            logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT is not set, skipping telemetry")
            self.is_otel_endpoint_set = False

        self.meter = metrics.get_meter(__name__)
        self._lock = _global_lock

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self.is_otel_endpoint_set:
            cast(TracerProvider, trace.get_tracer_provider()).force_flush()

    async def log_event(self, event: Event, ttl_seconds: int = 604800) -> None:
        if isinstance(event, UnstructuredLogEvent):
            self._log_unstructured(event, ttl_seconds)
        elif isinstance(event, MetricEvent):
            self._log_metric(event)
        elif isinstance(event, StructuredLogEvent):
            self._log_structured(event, ttl_seconds)
        else:
            raise ValueError(f"Unknown event type: {event}")

    def _log_unstructured(self, event: UnstructuredLogEvent, ttl_seconds: int) -> None:
        with self._lock:
            # Use global storage instead of instance storage
            span_id = int(event.span_id, 16)
            span = _GLOBAL_STORAGE["active_spans"].get(span_id)

            if span:
                timestamp_ns = int(event.timestamp.timestamp() * 1e9)
                span.add_event(
                    name=event.type.value,
                    attributes={
                        "message": event.message,
                        "severity": event.severity.value,
                        "__ttl__": ttl_seconds,
                        **(event.attributes or {}),
                    },
                    timestamp=timestamp_ns,
                )
            else:
                print(f"Warning: No active span found for span_id {span_id}. Dropping event: {event}")

    def _get_or_create_counter(self, name: str, unit: str) -> metrics.Counter:
        assert self.meter is not None
        if name not in _GLOBAL_STORAGE["counters"]:
            _GLOBAL_STORAGE["counters"][name] = self.meter.create_counter(
                name=name,
                unit=unit,
                description=f"Counter for {name}",
            )
        return cast(metrics.Counter, _GLOBAL_STORAGE["counters"][name])

    def _get_or_create_gauge(self, name: str, unit: str) -> metrics.ObservableGauge:
        assert self.meter is not None
        if name not in _GLOBAL_STORAGE["gauges"]:
            _GLOBAL_STORAGE["gauges"][name] = self.meter.create_gauge(
                name=name,
                unit=unit,
                description=f"Gauge for {name}",
            )
        return cast(metrics.ObservableGauge, _GLOBAL_STORAGE["gauges"][name])

    def _get_or_create_histogram(self, name: str, unit: str) -> metrics.Histogram:
        assert self.meter is not None
        if name not in _GLOBAL_STORAGE["histograms"]:
            _GLOBAL_STORAGE["histograms"][name] = self.meter.create_histogram(
                name=name,
                unit=unit,
                description=f"Histogram for {name}",
            )
        return cast(metrics.Histogram, _GLOBAL_STORAGE["histograms"][name])

    def _log_metric(self, event: MetricEvent) -> None:
        # Add metric as an event to the current span
        try:
            with self._lock:
                # Only try to add to span if we have a valid span_id
                if event.span_id:
                    try:
                        span_id = int(event.span_id, 16)
                        span = _GLOBAL_STORAGE["active_spans"].get(span_id)

                        if span:
                            timestamp_ns = int(event.timestamp.timestamp() * 1e9)
                            span.add_event(
                                name=f"metric.{event.metric}",
                                attributes={
                                    "value": event.value,
                                    "unit": event.unit,
                                    **(event.attributes or {}),
                                },
                                timestamp=timestamp_ns,
                            )
                    except (ValueError, KeyError):
                        # Invalid span_id or span not found, but we already logged to console above
                        pass
        except Exception:
            # Lock acquisition failed
            logger.debug("Failed to acquire lock to add metric to span")

        # Log to OpenTelemetry meter if available
        if self.meter is None:
            return

        # Use histograms for token-related metrics (per-request measurements)
        # Use counters for other cumulative metrics
        token_metrics = {"prompt_tokens", "completion_tokens", "total_tokens"}

        if event.metric in token_metrics:
            # Token metrics are per-request measurements, use histogram
            histogram = self._get_or_create_histogram(event.metric, event.unit)
            histogram.record(event.value, attributes=_clean_attributes(event.attributes))
        elif isinstance(event.value, int):
            counter = self._get_or_create_counter(event.metric, event.unit)
            counter.add(event.value, attributes=_clean_attributes(event.attributes))
        elif isinstance(event.value, float):
            up_down_counter = self._get_or_create_up_down_counter(event.metric, event.unit)
            up_down_counter.add(event.value, attributes=_clean_attributes(event.attributes))

    def _get_or_create_up_down_counter(self, name: str, unit: str) -> metrics.UpDownCounter:
        assert self.meter is not None
        if name not in _GLOBAL_STORAGE["up_down_counters"]:
            _GLOBAL_STORAGE["up_down_counters"][name] = self.meter.create_up_down_counter(
                name=name,
                unit=unit,
                description=f"UpDownCounter for {name}",
            )
        return cast(metrics.UpDownCounter, _GLOBAL_STORAGE["up_down_counters"][name])

    def _log_structured(self, event: StructuredLogEvent, ttl_seconds: int) -> None:
        with self._lock:
            span_id = int(event.span_id, 16)
            tracer = trace.get_tracer(__name__)
            if event.attributes is None:
                event.attributes = {}
            event.attributes["__ttl__"] = ttl_seconds

            # Extract these W3C trace context attributes so they are not written to
            # underlying storage, as we just need them to propagate the trace context.
            traceparent = event.attributes.pop("traceparent", None)
            tracestate = event.attributes.pop("tracestate", None)
            if traceparent:
                # If we have a traceparent header value, we're not the root span.
                for root_attribute in ROOT_SPAN_MARKERS:
                    event.attributes.pop(root_attribute, None)

            if isinstance(event.payload, SpanStartPayload):
                # Check if span already exists to prevent duplicates
                if span_id in _GLOBAL_STORAGE["active_spans"]:
                    return

                context = None
                if event.payload.parent_span_id:
                    parent_span_id = int(event.payload.parent_span_id, 16)
                    parent_span = _GLOBAL_STORAGE["active_spans"].get(parent_span_id)
                    if parent_span:
                        context = trace.set_span_in_context(parent_span)
                elif traceparent:
                    carrier = {
                        "traceparent": traceparent,
                        "tracestate": tracestate,
                    }
                    context = TraceContextTextMapPropagator().extract(carrier=carrier)

                span = tracer.start_span(
                    name=event.payload.name,
                    context=context,
                    attributes=_clean_attributes(event.attributes),
                )
                _GLOBAL_STORAGE["active_spans"][span_id] = span

            elif isinstance(event.payload, SpanEndPayload):
                span = _GLOBAL_STORAGE["active_spans"].get(span_id)  # type: ignore[assignment]
                if span:
                    if event.attributes:
                        cleaned_attrs = _clean_attributes(event.attributes)
                        if cleaned_attrs:
                            span.set_attributes(cleaned_attrs)

                    status = (
                        trace.Status(status_code=trace.StatusCode.OK)
                        if event.payload.status == SpanStatus.OK
                        else trace.Status(status_code=trace.StatusCode.ERROR)
                    )
                    span.set_status(status)
                    span.end()
                    _GLOBAL_STORAGE["active_spans"].pop(span_id, None)
            else:
                raise ValueError(f"Unknown structured log event: {event}")
