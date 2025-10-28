# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""In-memory telemetry collector for library-client tests."""

from typing import Any

import opentelemetry.metrics as otel_metrics
import opentelemetry.trace as otel_trace
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import llama_stack.core.telemetry.telemetry as telemetry_module

from .base import BaseTelemetryCollector, SpanStub


class InMemoryTelemetryCollector(BaseTelemetryCollector):
    def __init__(self, span_exporter: InMemorySpanExporter, metric_reader: InMemoryMetricReader) -> None:
        self._span_exporter = span_exporter
        self._metric_reader = metric_reader

    def _snapshot_spans(self) -> tuple[Any, ...]:
        spans = []
        for span in self._span_exporter.get_finished_spans():
            trace_id = None
            span_id = None
            context = getattr(span, "context", None)
            if context:
                trace_id = f"{context.trace_id:032x}"
                span_id = f"{context.span_id:016x}"
            else:
                trace_id = getattr(span, "trace_id", None)
                span_id = getattr(span, "span_id", None)

            stub = SpanStub(
                span.name,
                span.attributes,
                getattr(span, "resource", None),
                getattr(span, "events", None),
                trace_id,
                span_id,
            )
            spans.append(stub)

        return tuple(spans)

    def _snapshot_metrics(self) -> Any | None:
        data = self._metric_reader.get_metrics_data()
        if data and data.resource_metrics:
            resource_metric = data.resource_metrics[0]
            if resource_metric.scope_metrics:
                return resource_metric.scope_metrics[0].metrics
        return None

    def _clear_impl(self) -> None:
        self._span_exporter.clear()
        self._metric_reader.get_metrics_data()


class InMemoryTelemetryManager:
    def __init__(self) -> None:
        if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
            otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        if hasattr(otel_metrics, "_METER_PROVIDER_SET_ONCE"):
            otel_metrics._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        telemetry_module._TRACER_PROVIDER = tracer_provider

        self.collector = InMemoryTelemetryCollector(span_exporter, metric_reader)
        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider

    def shutdown(self) -> None:
        telemetry_module._TRACER_PROVIDER = None
        self._tracer_provider.shutdown()
        self._meter_provider.shutdown()
