# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry test configuration using OpenTelemetry SDK exporters.

This conftest provides in-memory telemetry collection for library_client mode only.
Tests using these fixtures should skip in server mode since the in-memory collector
cannot access spans from a separate server process.
"""

from typing import Any

import opentelemetry.trace as otel_trace
import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import llama_stack.providers.inline.telemetry.meta_reference.telemetry as telemetry_module


class OtelTestCollector:
    """In-memory collector for OpenTelemetry traces and metrics."""

    def __init__(self):
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        trace.set_tracer_provider(self.tracer_provider)

        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(metric_readers=[self.metric_reader])
        metrics.set_meter_provider(self.meter_provider)

    def get_spans(self) -> tuple[ReadableSpan, ...]:
        return self.span_exporter.get_finished_spans()

    def get_metrics(self) -> Any | None:
        return self.metric_reader.get_metrics_data()

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


@pytest.fixture
def mock_otlp_collector():
    """Function-scoped: Fresh telemetry data view for each test."""
    if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore

    collector = OtelTestCollector()
    telemetry_module._TRACER_PROVIDER = collector.tracer_provider

    yield collector

    telemetry_module._TRACER_PROVIDER = None
    collector.shutdown()
