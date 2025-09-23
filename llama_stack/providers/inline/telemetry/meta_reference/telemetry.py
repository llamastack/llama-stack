# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import datetime
import threading
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from llama_stack.apis.telemetry import (
    Event,
    MetricEvent,
    MetricLabelMatcher,
    MetricQueryType,
    QueryCondition,
    QueryMetricsResponse,
    QuerySpanTreeResponse,
    QueryTracesResponse,
    Span,
    SpanEndPayload,
    SpanStartPayload,
    SpanStatus,
    StructuredLogEvent,
    Telemetry,
    Trace,
    UnstructuredLogEvent,
)
from llama_stack.core.datatypes import Api
from llama_stack.log import get_logger
from llama_stack.providers.inline.telemetry.meta_reference.console_span_processor import (
    ConsoleSpanProcessor,
)
from llama_stack.providers.inline.telemetry.meta_reference.sqlite_span_processor import (
    SQLiteSpanProcessor,
)
from llama_stack.providers.utils.telemetry.dataset_mixin import TelemetryDatasetMixin
from llama_stack.providers.utils.telemetry.sqlite_trace_store import SQLiteTraceStore
from llama_stack.providers.utils.telemetry.tracing import ROOT_SPAN_MARKERS

from .config import TelemetryConfig, TelemetrySink

_GLOBAL_STORAGE: dict[str, dict[str | int, Any]] = {
    "active_spans": {},
    "counters": {},
    "gauges": {},
    "up_down_counters": {},
}
_global_lock = threading.Lock()
_TRACER_PROVIDER = None

logger = get_logger(name=__name__, category="telemetry")


def is_tracing_enabled(tracer):
    with tracer.start_as_current_span("check_tracing") as span:
        return span.is_recording()


class TelemetryAdapter(TelemetryDatasetMixin, Telemetry):
    def __init__(self, config: TelemetryConfig, deps: dict[Api, Any]) -> None:
        self.config = config
        self.datasetio_api = deps.get(Api.datasetio)
        self.meter = None

        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
            }
        )

        global _TRACER_PROVIDER
        # Initialize the correct span processor based on the provider state.
        # This is needed since once the span processor is set, it cannot be unset.
        # Recreating the telemetry adapter multiple times will result in duplicate span processors.
        # Since the library client can be recreated multiple times in a notebook,
        # the kernel will hold on to the span processor and cause duplicate spans to be written.
        if _TRACER_PROVIDER is None:
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            _TRACER_PROVIDER = provider

            # Use single OTLP endpoint for all telemetry signals
            if TelemetrySink.OTEL_TRACE in self.config.sinks or TelemetrySink.OTEL_METRIC in self.config.sinks:
                if self.config.otel_exporter_otlp_endpoint is None:
                    raise ValueError(
                        "otel_exporter_otlp_endpoint is required when OTEL_TRACE or OTEL_METRIC is enabled"
                    )

                # Let OpenTelemetry SDK handle endpoint construction automatically
                # The SDK will read OTEL_EXPORTER_OTLP_ENDPOINT and construct appropriate URLs
                # https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter
                if TelemetrySink.OTEL_TRACE in self.config.sinks:
                    span_exporter = OTLPSpanExporter()
                    span_processor = BatchSpanProcessor(span_exporter)
                    trace.get_tracer_provider().add_span_processor(span_processor)

                if TelemetrySink.OTEL_METRIC in self.config.sinks:
                    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
                    metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                    metrics.set_meter_provider(metric_provider)

            if TelemetrySink.SQLITE in self.config.sinks:
                trace.get_tracer_provider().add_span_processor(SQLiteSpanProcessor(self.config.sqlite_db_path))
            if TelemetrySink.CONSOLE in self.config.sinks:
                trace.get_tracer_provider().add_span_processor(ConsoleSpanProcessor(print_attributes=True))

        if TelemetrySink.OTEL_METRIC in self.config.sinks:
            self.meter = metrics.get_meter(__name__)
        if TelemetrySink.SQLITE in self.config.sinks:
            self.trace_store = SQLiteTraceStore(self.config.sqlite_db_path)

        self._lock = _global_lock

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        trace.get_tracer_provider().force_flush()

    async def log_event(self, event: Event, ttl_seconds: int = 604800) -> None:
        logger.debug(f"DEBUG: log_event called with event type: {type(event).__name__}")
        if isinstance(event, UnstructuredLogEvent):
            self._log_unstructured(event, ttl_seconds)
        elif isinstance(event, MetricEvent):
            logger.debug("DEBUG: Routing MetricEvent to _log_metric")
            self._log_metric(event)
        elif isinstance(event, StructuredLogEvent):
            self._log_structured(event, ttl_seconds)
        else:
            raise ValueError(f"Unknown event type: {event}")

    async def query_metrics(
        self,
        metric_name: str,
        start_time: int,
        end_time: int | None = None,
        granularity: str | None = None,
        query_type: MetricQueryType = MetricQueryType.RANGE,
        label_matchers: list[MetricLabelMatcher] | None = None,
    ) -> QueryMetricsResponse:
        """Query metrics from the telemetry store.

        Args:
            metric_name: The name of the metric to query (e.g., "prompt_tokens")
            start_time: Start time as Unix timestamp
            end_time: End time as Unix timestamp (defaults to now if None)
            granularity: Time granularity for aggregation
            query_type: Type of query (RANGE or INSTANT)
            label_matchers: Label filters to apply

        Returns:
            QueryMetricsResponse with metric time series data
        """
        # Convert timestamps to datetime objects
        start_dt = datetime.datetime.fromtimestamp(start_time, datetime.UTC)
        end_dt = datetime.datetime.fromtimestamp(end_time, datetime.UTC) if end_time else None

        # Use SQLite trace store if available
        if hasattr(self, "trace_store") and self.trace_store:
            return await self.trace_store.query_metrics(
                metric_name=metric_name,
                start_time=start_dt,
                end_time=end_dt,
                granularity=granularity,
                query_type=query_type,
                label_matchers=label_matchers,
            )
        else:
            raise ValueError(
                f"In order to query_metrics, you must have {TelemetrySink.SQLITE} set in your telemetry sinks"
            )

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
        return _GLOBAL_STORAGE["counters"][name]

    def _get_or_create_gauge(self, name: str, unit: str) -> metrics.ObservableGauge:
        assert self.meter is not None
        if name not in _GLOBAL_STORAGE["gauges"]:
            _GLOBAL_STORAGE["gauges"][name] = self.meter.create_gauge(
                name=name,
                unit=unit,
                description=f"Gauge for {name}",
            )
        return _GLOBAL_STORAGE["gauges"][name]

    def _log_metric(self, event: MetricEvent) -> None:
        # Always log to console if console sink is enabled (debug)
        if TelemetrySink.CONSOLE in self.config.sinks:
            logger.debug(f"METRIC: {event.metric}={event.value} {event.unit} {event.attributes}")

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
        if isinstance(event.value, int):
            counter = self._get_or_create_counter(event.metric, event.unit)
            counter.add(event.value, attributes=event.attributes)
        elif isinstance(event.value, float):
            up_down_counter = self._get_or_create_up_down_counter(event.metric, event.unit)
            up_down_counter.add(event.value, attributes=event.attributes)

    def _get_or_create_up_down_counter(self, name: str, unit: str) -> metrics.UpDownCounter:
        assert self.meter is not None
        if name not in _GLOBAL_STORAGE["up_down_counters"]:
            _GLOBAL_STORAGE["up_down_counters"][name] = self.meter.create_up_down_counter(
                name=name,
                unit=unit,
                description=f"UpDownCounter for {name}",
            )
        return _GLOBAL_STORAGE["up_down_counters"][name]

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
                    attributes=event.attributes or {},
                )
                _GLOBAL_STORAGE["active_spans"][span_id] = span

            elif isinstance(event.payload, SpanEndPayload):
                span = _GLOBAL_STORAGE["active_spans"].get(span_id)
                if span:
                    if event.attributes:
                        span.set_attributes(event.attributes)

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

    async def query_traces(
        self,
        attribute_filters: list[QueryCondition] | None = None,
        limit: int | None = 100,
        offset: int | None = 0,
        order_by: list[str] | None = None,
    ) -> QueryTracesResponse:
        return QueryTracesResponse(
            data=await self.trace_store.query_traces(
                attribute_filters=attribute_filters,
                limit=limit,
                offset=offset,
                order_by=order_by,
            )
        )

    async def get_trace(self, trace_id: str) -> Trace:
        return await self.trace_store.get_trace(trace_id)

    async def get_span(self, trace_id: str, span_id: str) -> Span:
        return await self.trace_store.get_span(trace_id, span_id)

    async def get_span_tree(
        self,
        span_id: str,
        attributes_to_return: list[str] | None = None,
        max_depth: int | None = None,
    ) -> QuerySpanTreeResponse:
        return QuerySpanTreeResponse(
            data=await self.trace_store.get_span_tree(
                span_id=span_id,
                attributes_to_return=attributes_to_return,
                max_depth=max_depth,
            )
        )
