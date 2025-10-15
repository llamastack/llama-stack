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
from llama_stack.testing.api_recorder import patch_httpx_for_test_id
from tests.integration.fixtures.common import instantiate_llama_stack_client


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

    def clear(self) -> None:
        self.span_exporter.clear()
        self.metric_reader.get_metrics_data()


@pytest.fixture(scope="session")
def _telemetry_providers():
    """Set up in-memory OTEL providers before llama_stack_client initializes."""
    # Reset set-once flags to allow re-initialization
    if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore
    if hasattr(otel_metrics, "_METER_PROVIDER_SET_ONCE"):
        otel_metrics._METER_PROVIDER_SET_ONCE._done = False  # type: ignore

    # Create in-memory exporters/readers
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Set module-level provider so TelemetryAdapter uses our in-memory providers
    telemetry_module._TRACER_PROVIDER = tracer_provider

    yield (span_exporter, metric_reader, tracer_provider, meter_provider)

    telemetry_module._TRACER_PROVIDER = None
    tracer_provider.shutdown()
    meter_provider.shutdown()


@pytest.fixture(scope="session")
def llama_stack_client(_telemetry_providers, request):
    """Override llama_stack_client to ensure in-memory telemetry providers are used."""
    patch_httpx_for_test_id()
    client = instantiate_llama_stack_client(request.session)

    return client


@pytest.fixture
def mock_otlp_collector(_telemetry_providers):
    """Provides access to telemetry data and clears between tests."""
    span_exporter, metric_reader, _, _ = _telemetry_providers
    collector = TestCollector(span_exporter, metric_reader)
    yield collector
    collector.clear()
