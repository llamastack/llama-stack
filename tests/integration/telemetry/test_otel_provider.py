# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for OpenTelemetry provider.

These tests verify that the OTel provider correctly:
- Initializes within the Llama Stack
- Captures expected metrics (counters, histograms, up/down counters)
- Captures expected spans/traces
- Exports telemetry data to an OTLP collector (in-memory for testing)

Tests use in-memory exporters to avoid external dependencies and can run in GitHub Actions.
"""

import os
import time
from collections import defaultdict
from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llama_stack.providers.inline.telemetry.otel.config import OTelTelemetryConfig
from llama_stack.providers.inline.telemetry.otel.otel import OTelTelemetryProvider


@pytest.fixture(scope="module")
def in_memory_span_exporter():
    """Create an in-memory span exporter to capture traces."""
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def in_memory_metric_reader():
    """Create an in-memory metric reader to capture metrics."""
    return InMemoryMetricReader()


@pytest.fixture(scope="module")
def otel_provider_with_memory_exporters(in_memory_span_exporter, in_memory_metric_reader):
    """
    Create an OTelTelemetryProvider configured with in-memory exporters.
    
    This allows us to capture and verify telemetry data without external services.
    Returns a dict with 'provider', 'span_exporter', and 'metric_reader'.
    """
    # Set mock environment to avoid warnings
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    
    config = OTelTelemetryConfig(
        service_name="test-llama-stack-otel",
        service_version="1.0.0-test",
        deployment_environment="ci-test",
        span_processor="simple",
    )
    
    # Patch the provider to use in-memory exporters
    with patch.object(
        OTelTelemetryProvider,
        'model_post_init',
        lambda self, _: _init_with_memory_exporters(
            self, config, in_memory_span_exporter, in_memory_metric_reader
        )
    ):
        provider = OTelTelemetryProvider(config=config)
        yield {
            'provider': provider,
            'span_exporter': in_memory_span_exporter,
            'metric_reader': in_memory_metric_reader
        }


def _init_with_memory_exporters(provider, config, span_exporter, metric_reader):
    """Helper to initialize provider with in-memory exporters."""
    import threading
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Attributes, Resource
    from opentelemetry.sdk.trace import TracerProvider
    
    # Initialize pydantic private attributes
    if provider.__pydantic_private__ is None:
        provider.__pydantic_private__ = {}
    
    provider._lock = threading.Lock()
    provider._counters = {}
    provider._up_down_counters = {}
    provider._histograms = {}
    provider._gauges = {}
    
    # Create resource attributes
    attributes: Attributes = {
        key: value
        for key, value in {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.deployment_environment,
        }.items()
        if value is not None
    }
    
    resource = Resource.create(attributes)
    
    # Configure tracer provider with in-memory exporter
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    
    # Configure meter provider with in-memory reader
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    metrics.set_meter_provider(meter_provider)


class TestOTelProviderInitialization:
    """Test OTel provider initialization within Llama Stack."""
    
    def test_provider_initializes_successfully(self, otel_provider_with_memory_exporters):
        """Test that the OTel provider initializes without errors."""
        provider = otel_provider_with_memory_exporters['provider']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        assert provider is not None
        assert provider.config.service_name == "test-llama-stack-otel"
        assert provider.config.service_version == "1.0.0-test"
        assert provider.config.deployment_environment == "ci-test"
    
    def test_provider_has_thread_safety_mechanisms(self, otel_provider_with_memory_exporters):
        """Test that the provider has thread-safety mechanisms in place."""
        provider = otel_provider_with_memory_exporters['provider']
        
        assert hasattr(provider, "_lock")
        assert provider._lock is not None
        assert hasattr(provider, "_counters")
        assert hasattr(provider, "_histograms")
        assert hasattr(provider, "_up_down_counters")


class TestOTelMetricsCapture:
    """Test that OTel provider captures expected metrics."""
    
    def test_counter_metric_is_captured(self, otel_provider_with_memory_exporters):
        """Test that counter metrics are captured."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        
        # Record counter metrics
        provider.record_count("llama.requests.total", 1.0, attributes={"endpoint": "/chat"})
        provider.record_count("llama.requests.total", 1.0, attributes={"endpoint": "/chat"})
        provider.record_count("llama.requests.total", 1.0, attributes={"endpoint": "/embeddings"})
        
        # Force metric collection - collect() triggers the reader to gather metrics
        metric_reader.collect()
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        
        # Verify metrics were captured
        assert metrics_data is not None
        assert len(metrics_data.resource_metrics) > 0
        
        # Find our counter metric
        found_counter = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "llama.requests.total":
                        found_counter = True
                        # Verify it's a counter with data points
                        assert hasattr(metric.data, "data_points")
                        assert len(metric.data.data_points) > 0
        
        assert found_counter, "Counter metric 'llama.requests.total' was not captured"
    
    def test_histogram_metric_is_captured(self, otel_provider_with_memory_exporters):
        """Test that histogram metrics are captured."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        
        # Record histogram metrics with various values
        latencies = [10.5, 25.3, 50.1, 100.7, 250.2]
        for latency in latencies:
            provider.record_histogram(
                "llama.inference.latency",
                latency,
                attributes={"model": "llama-3.2"}
            )
        
        # Force metric collection
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        
        # Find our histogram metric
        found_histogram = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "llama.inference.latency":
                        found_histogram = True
                        # Verify it's a histogram
                        assert hasattr(metric.data, "data_points")
                        data_point = metric.data.data_points[0]
                        # Histograms should have count and sum
                        assert hasattr(data_point, "count")
                        assert data_point.count == len(latencies)
        
        assert found_histogram, "Histogram metric 'llama.inference.latency' was not captured"
    
    def test_up_down_counter_metric_is_captured(self, otel_provider_with_memory_exporters):
        """Test that up/down counter metrics are captured."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        
        # Record up/down counter metrics
        provider.record_up_down_counter("llama.active.sessions", 5)
        provider.record_up_down_counter("llama.active.sessions", 3)
        provider.record_up_down_counter("llama.active.sessions", -2)
        
        # Force metric collection
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        
        # Find our up/down counter metric
        found_updown = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "llama.active.sessions":
                        found_updown = True
                        assert hasattr(metric.data, "data_points")
                        assert len(metric.data.data_points) > 0
        
        assert found_updown, "Up/Down counter metric 'llama.active.sessions' was not captured"
    
    def test_metrics_with_attributes_are_captured(self, otel_provider_with_memory_exporters):
        """Test that metric attributes/labels are preserved."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        
        # Record metrics with different attributes
        provider.record_count("llama.tokens.generated", 150.0, attributes={
            "model": "llama-3.2-1b",
            "user": "test-user"
        })
        
        # Force metric collection
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        
        # Verify attributes are preserved
        found_with_attributes = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "llama.tokens.generated":
                        data_point = metric.data.data_points[0]
                        # Check attributes - they're already a dict in the SDK
                        attrs = data_point.attributes if isinstance(data_point.attributes, dict) else {}
                        if "model" in attrs and "user" in attrs:
                            found_with_attributes = True
                            assert attrs["model"] == "llama-3.2-1b"
                            assert attrs["user"] == "test-user"
        
        assert found_with_attributes, "Metrics with attributes were not properly captured"
    
    def test_multiple_metric_types_coexist(self, otel_provider_with_memory_exporters):
        """Test that different metric types can coexist."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        
        # Record various metric types
        provider.record_count("test.counter", 1.0)
        provider.record_histogram("test.histogram", 42.0)
        provider.record_up_down_counter("test.gauge", 10)
        
        # Force metric collection
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        
        # Count unique metrics
        metric_names = set()
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metric_names.add(metric.name)
        
        # Should have all three metrics
        assert "test.counter" in metric_names
        assert "test.histogram" in metric_names
        assert "test.gauge" in metric_names


class TestOTelSpansCapture:
    """Test that OTel provider captures expected spans/traces."""
    
    def test_basic_span_is_captured(self, otel_provider_with_memory_exporters):
        """Test that basic spans are captured."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        # Create a span
        span = provider.custom_trace("llama.inference.request")
        span.end()
        
        # Get captured spans
        spans = span_exporter.get_finished_spans()
        
        assert len(spans) > 0
        assert any(span.name == "llama.inference.request" for span in spans)
    
    def test_span_with_attributes_is_captured(self, otel_provider_with_memory_exporters):
        """Test that span attributes are preserved."""
        provider = otel_provider_with_memory_exporters['provider']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        # Create a span with attributes
        span = provider.custom_trace(
            "llama.chat.completion",
            attributes={
                "model.id": "llama-3.2-1b",
                "user.id": "test-user-123",
                "request.id": "req-abc-123"
            }
        )
        span.end()
        
        # Get captured spans
        spans = span_exporter.get_finished_spans()
        
        # Find our span
        our_span = None
        for s in spans:
            if s.name == "llama.chat.completion":
                our_span = s
                break
        
        assert our_span is not None, "Span 'llama.chat.completion' was not captured"
        
        # Verify attributes
        attrs = dict(our_span.attributes)
        assert attrs.get("model.id") == "llama-3.2-1b"
        assert attrs.get("user.id") == "test-user-123"
        assert attrs.get("request.id") == "req-abc-123"
    
    def test_multiple_spans_are_captured(self, otel_provider_with_memory_exporters):
        """Test that multiple spans are captured."""
        provider = otel_provider_with_memory_exporters['provider']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        # Create multiple spans
        span_names = [
            "llama.request.validate",
            "llama.model.load",
            "llama.inference.execute",
            "llama.response.format"
        ]
        
        for name in span_names:
            span = provider.custom_trace(name)
            time.sleep(0.01)  # Small delay to ensure ordering
            span.end()
        
        # Get captured spans
        spans = span_exporter.get_finished_spans()
        captured_names = {span.name for span in spans}
        
        # Verify all spans were captured
        for expected_name in span_names:
            assert expected_name in captured_names, f"Span '{expected_name}' was not captured"
    
    def test_span_has_service_metadata(self, otel_provider_with_memory_exporters):
        """Test that spans include service metadata."""
        provider = otel_provider_with_memory_exporters['provider']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        # Create a span
        span = provider.custom_trace("test.span")
        span.end()
        
        # Get captured spans
        spans = span_exporter.get_finished_spans()
        
        assert len(spans) > 0
        
        # Check resource attributes
        span = spans[0]
        resource_attrs = dict(span.resource.attributes)
        
        assert resource_attrs.get("service.name") == "test-llama-stack-otel"
        assert resource_attrs.get("service.version") == "1.0.0-test"
        assert resource_attrs.get("deployment.environment") == "ci-test"


class TestOTelDataExport:
    """Test that telemetry data can be exported to OTLP collector."""
    
    def test_metrics_are_exportable(self, otel_provider_with_memory_exporters):
        """Test that metrics can be exported."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        
        # Record metrics
        provider.record_count("export.test.counter", 5.0)
        provider.record_histogram("export.test.histogram", 123.45)
        
        # Force export
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        
        # Verify data structure is exportable
        assert metrics_data is not None
        assert hasattr(metrics_data, "resource_metrics")
        assert len(metrics_data.resource_metrics) > 0
        
        # Verify resource attributes are present (needed for OTLP export)
        resource = metrics_data.resource_metrics[0].resource
        assert resource is not None
        assert len(resource.attributes) > 0
    
    def test_spans_are_exportable(self, otel_provider_with_memory_exporters):
        """Test that spans can be exported."""
        provider = otel_provider_with_memory_exporters['provider']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        # Create spans
        span1 = provider.custom_trace("export.test.span1")
        span1.end()
        
        span2 = provider.custom_trace("export.test.span2")
        span2.end()
        
        # Get exported spans
        spans = span_exporter.get_finished_spans()
        
        # Verify spans have required OTLP fields
        assert len(spans) >= 2
        for span in spans:
            assert span.name is not None
            assert span.context is not None
            assert span.context.trace_id is not None
            assert span.context.span_id is not None
            assert span.resource is not None
    
    def test_concurrent_export_is_safe(self, otel_provider_with_memory_exporters):
        """Test that concurrent metric/span recording doesn't break export."""
        import concurrent.futures
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        def record_data(thread_id):
            for i in range(10):
                provider.record_count(f"concurrent.counter.{thread_id}", 1.0)
                span = provider.custom_trace(f"concurrent.span.{thread_id}.{i}")
                span.end()
        
        # Record from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(record_data, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # Verify export still works
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        spans = span_exporter.get_finished_spans()
        
        assert metrics_data is not None
        assert len(spans) >= 50  # 5 threads * 10 spans each


@pytest.mark.integration
class TestOTelProviderIntegration:
    """End-to-end integration tests simulating real usage."""
    
    def test_complete_inference_workflow_telemetry(self, otel_provider_with_memory_exporters):
        """Simulate a complete inference workflow with telemetry."""
        provider = otel_provider_with_memory_exporters['provider']
        metric_reader = otel_provider_with_memory_exporters['metric_reader']
        span_exporter = otel_provider_with_memory_exporters['span_exporter']
        
        # Simulate inference workflow
        request_span = provider.custom_trace(
            "llama.inference.request",
            attributes={"model": "llama-3.2-1b", "user": "test"}
        )
        
        # Track metrics during inference
        provider.record_count("llama.requests.received", 1.0)
        provider.record_up_down_counter("llama.requests.in_flight", 1)
        
        # Simulate processing time
        time.sleep(0.01)
        provider.record_histogram("llama.request.duration_ms", 10.5)
        
        # Track tokens
        provider.record_count("llama.tokens.input", 25.0)
        provider.record_count("llama.tokens.output", 150.0)
        
        # End request
        provider.record_up_down_counter("llama.requests.in_flight", -1)
        provider.record_count("llama.requests.completed", 1.0)
        request_span.end()
        
        # Verify all telemetry was captured
        metric_reader.collect()
        metrics_data = metric_reader.get_metrics_data()
        spans = span_exporter.get_finished_spans()
        
        # Check metrics exist
        metric_names = set()
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    metric_names.add(m.name)
        
        assert "llama.requests.received" in metric_names
        assert "llama.requests.in_flight" in metric_names
        assert "llama.request.duration_ms" in metric_names
        assert "llama.tokens.input" in metric_names
        assert "llama.tokens.output" in metric_names
        
        # Check span exists
        assert any(s.name == "llama.inference.request" for s in spans)

