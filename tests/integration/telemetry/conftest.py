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

import opentelemetry.metrics as otel_metrics
import opentelemetry.trace as otel_trace
import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import llama_stack.providers.inline.telemetry.meta_reference.telemetry as telemetry_module


@pytest.fixture(scope="session")
def _setup_test_telemetry():
    """Session-scoped: Set up test telemetry providers before client initialization."""
    # Reset OpenTelemetry's internal locks to allow test fixtures to override providers
    if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore
    if hasattr(otel_metrics, "_METER_PROVIDER_SET_ONCE"):
        otel_metrics._METER_PROVIDER_SET_ONCE._done = False  # type: ignore

    # Create and set up providers before client initialization
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Set module-level providers so TelemetryAdapter uses them
    telemetry_module._TRACER_PROVIDER = tracer_provider

    yield tracer_provider, meter_provider, span_exporter, metric_reader

    # Cleanup
    telemetry_module._TRACER_PROVIDER = None
    tracer_provider.shutdown()
    meter_provider.shutdown()


class TestCollector:
    def __init__(self, span_exp, metric_read):
        assert span_exp and metric_read
        self.span_exporter = span_exp
        self.metric_reader = metric_read

    def get_spans(self) -> tuple[ReadableSpan, ...]:
        return self.span_exporter.get_finished_spans()

    def get_metrics(self) -> Any | None:
        metrics = self.metric_reader.get_metrics_data()
        if metrics and metrics.resource_metrics:
            return metrics.resource_metrics[0].scope_metrics[0].metrics
        return None


@pytest.fixture
def mock_otlp_collector(_setup_test_telemetry):
    """Function-scoped: Access to telemetry data for each test."""
    # Unpack the providers from the session fixture
    tracer_provider, meter_provider, span_exporter, metric_reader = _setup_test_telemetry

    collector = TestCollector(span_exporter, metric_reader)

    # Clear spans between tests
    span_exporter.clear()

    yield collector
