# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import concurrent.futures
import threading
from unittest.mock import MagicMock

import pytest

from llama_stack.providers.inline.telemetry.otel.config import OTelTelemetryConfig
from llama_stack.providers.inline.telemetry.otel.otel import OTelTelemetryProvider


@pytest.fixture
def otel_config():
    """Fixture providing a basic OTelTelemetryConfig."""
    return OTelTelemetryConfig(
        service_name="test-service",
        service_version="1.0.0",
        deployment_environment="test",
        span_processor="simple",
    )


@pytest.fixture
def otel_provider(otel_config, monkeypatch):
    """Fixture providing an OTelTelemetryProvider instance with mocked environment."""
    # Set required environment variables to avoid warnings
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    return OTelTelemetryProvider(config=otel_config)


class TestOTelTelemetryProviderInitialization:
    """Tests for OTelTelemetryProvider initialization."""

    def test_initialization_with_valid_config(self, otel_config, monkeypatch):
        """Test that provider initializes correctly with valid configuration."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        
        provider = OTelTelemetryProvider(config=otel_config)
        
        assert provider.config == otel_config
        assert hasattr(provider, "_lock")
        assert provider._lock is not None
        assert isinstance(provider._counters, dict)
        assert isinstance(provider._histograms, dict)
        assert isinstance(provider._up_down_counters, dict)
        assert isinstance(provider._gauges, dict)

    def test_initialization_sets_service_attributes(self, otel_config, monkeypatch):
        """Test that service attributes are properly configured."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        
        provider = OTelTelemetryProvider(config=otel_config)
        
        assert provider.config.service_name == "test-service"
        assert provider.config.service_version == "1.0.0"
        assert provider.config.deployment_environment == "test"

    def test_initialization_with_batch_processor(self, monkeypatch):
        """Test initialization with batch span processor."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        config = OTelTelemetryConfig(
            service_name="test-service",
            service_version="1.0.0",
            deployment_environment="test",
            span_processor="batch",
        )
        
        provider = OTelTelemetryProvider(config=config)
        
        assert provider.config.span_processor == "batch"

    def test_warns_when_endpoints_missing(self, otel_config, monkeypatch, caplog):
        """Test that warnings are issued when OTLP endpoints are not set."""
        # Remove all endpoint environment variables
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
        
        OTelTelemetryProvider(config=otel_config)
        
        # Check that warnings were logged
        assert any("Traces will not be exported" in record.message for record in caplog.records)
        assert any("Metrics will not be exported" in record.message for record in caplog.records)


class TestOTelTelemetryProviderMetrics:
    """Tests for metric recording functionality."""

    def test_record_count_creates_counter(self, otel_provider):
        """Test that record_count creates a counter on first call."""
        assert "test_counter" not in otel_provider._counters
        
        otel_provider.record_count("test_counter", 1.0)
        
        assert "test_counter" in otel_provider._counters
        assert otel_provider._counters["test_counter"] is not None

    def test_record_count_reuses_counter(self, otel_provider):
        """Test that record_count reuses existing counter."""
        otel_provider.record_count("test_counter", 1.0)
        first_counter = otel_provider._counters["test_counter"]
        
        otel_provider.record_count("test_counter", 2.0)
        second_counter = otel_provider._counters["test_counter"]
        
        assert first_counter is second_counter
        assert len(otel_provider._counters) == 1

    def test_record_count_with_attributes(self, otel_provider):
        """Test that record_count works with attributes."""
        otel_provider.record_count(
            "test_counter",
            1.0,
            attributes={"key": "value", "env": "test"}
        )
        
        assert "test_counter" in otel_provider._counters

    def test_record_histogram_creates_histogram(self, otel_provider):
        """Test that record_histogram creates a histogram on first call."""
        assert "test_histogram" not in otel_provider._histograms
        
        otel_provider.record_histogram("test_histogram", 42.5)
        
        assert "test_histogram" in otel_provider._histograms
        assert otel_provider._histograms["test_histogram"] is not None

    def test_record_histogram_reuses_histogram(self, otel_provider):
        """Test that record_histogram reuses existing histogram."""
        otel_provider.record_histogram("test_histogram", 10.0)
        first_histogram = otel_provider._histograms["test_histogram"]
        
        otel_provider.record_histogram("test_histogram", 20.0)
        second_histogram = otel_provider._histograms["test_histogram"]
        
        assert first_histogram is second_histogram
        assert len(otel_provider._histograms) == 1

    def test_record_histogram_with_bucket_boundaries(self, otel_provider):
        """Test that record_histogram works with explicit bucket boundaries."""
        boundaries = [0.0, 10.0, 50.0, 100.0]
        
        otel_provider.record_histogram(
            "test_histogram",
            25.0,
            explicit_bucket_boundaries_advisory=boundaries
        )
        
        assert "test_histogram" in otel_provider._histograms

    def test_record_up_down_counter_creates_counter(self, otel_provider):
        """Test that record_up_down_counter creates a counter on first call."""
        assert "test_updown" not in otel_provider._up_down_counters
        
        otel_provider.record_up_down_counter("test_updown", 1.0)
        
        assert "test_updown" in otel_provider._up_down_counters
        assert otel_provider._up_down_counters["test_updown"] is not None

    def test_record_up_down_counter_reuses_counter(self, otel_provider):
        """Test that record_up_down_counter reuses existing counter."""
        otel_provider.record_up_down_counter("test_updown", 5.0)
        first_counter = otel_provider._up_down_counters["test_updown"]
        
        otel_provider.record_up_down_counter("test_updown", -3.0)
        second_counter = otel_provider._up_down_counters["test_updown"]
        
        assert first_counter is second_counter
        assert len(otel_provider._up_down_counters) == 1

    def test_multiple_metrics_with_different_names(self, otel_provider):
        """Test that multiple metrics with different names are cached separately."""
        otel_provider.record_count("counter1", 1.0)
        otel_provider.record_count("counter2", 2.0)
        otel_provider.record_histogram("histogram1", 10.0)
        otel_provider.record_up_down_counter("updown1", 5.0)
        
        assert len(otel_provider._counters) == 2
        assert len(otel_provider._histograms) == 1
        assert len(otel_provider._up_down_counters) == 1


class TestOTelTelemetryProviderThreadSafety:
    """Tests for thread safety of metric operations."""

    def test_concurrent_counter_creation_same_name(self, otel_provider):
        """Test that concurrent calls to record_count with same name are thread-safe."""
        num_threads = 50
        counter_name = "concurrent_counter"
        
        def record_metric():
            otel_provider.record_count(counter_name, 1.0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_metric) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Should have exactly one counter created despite concurrent access
        assert len(otel_provider._counters) == 1
        assert counter_name in otel_provider._counters

    def test_concurrent_histogram_creation_same_name(self, otel_provider):
        """Test that concurrent calls to record_histogram with same name are thread-safe."""
        num_threads = 50
        histogram_name = "concurrent_histogram"
        
        def record_metric():
            thread_id = threading.current_thread().ident or 0
            otel_provider.record_histogram(histogram_name, float(thread_id % 100))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_metric) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Should have exactly one histogram created despite concurrent access
        assert len(otel_provider._histograms) == 1
        assert histogram_name in otel_provider._histograms

    def test_concurrent_up_down_counter_creation_same_name(self, otel_provider):
        """Test that concurrent calls to record_up_down_counter with same name are thread-safe."""
        num_threads = 50
        counter_name = "concurrent_updown"
        
        def record_metric():
            otel_provider.record_up_down_counter(counter_name, 1.0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_metric) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Should have exactly one counter created despite concurrent access
        assert len(otel_provider._up_down_counters) == 1
        assert counter_name in otel_provider._up_down_counters

    def test_concurrent_mixed_metrics_different_names(self, otel_provider):
        """Test concurrent creation of different metric types with different names."""
        num_threads = 30
        
        def record_counters(thread_id):
            otel_provider.record_count(f"counter_{thread_id}", 1.0)
        
        def record_histograms(thread_id):
            otel_provider.record_histogram(f"histogram_{thread_id}", float(thread_id))
        
        def record_up_down_counters(thread_id):
            otel_provider.record_up_down_counter(f"updown_{thread_id}", float(thread_id))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads * 3) as executor:
            futures = []
            for i in range(num_threads):
                futures.append(executor.submit(record_counters, i))
                futures.append(executor.submit(record_histograms, i))
                futures.append(executor.submit(record_up_down_counters, i))
            
            concurrent.futures.wait(futures)
        
        # Each thread should have created its own metric
        assert len(otel_provider._counters) == num_threads
        assert len(otel_provider._histograms) == num_threads
        assert len(otel_provider._up_down_counters) == num_threads

    def test_concurrent_access_existing_and_new_metrics(self, otel_provider):
        """Test concurrent access mixing existing and new metric creation."""
        # Pre-create some metrics
        otel_provider.record_count("existing_counter", 1.0)
        otel_provider.record_histogram("existing_histogram", 10.0)
        
        num_threads = 40
        
        def mixed_operations(thread_id):
            # Half the threads use existing metrics, half create new ones
            if thread_id % 2 == 0:
                otel_provider.record_count("existing_counter", 1.0)
                otel_provider.record_histogram("existing_histogram", float(thread_id))
            else:
                otel_provider.record_count(f"new_counter_{thread_id}", 1.0)
                otel_provider.record_histogram(f"new_histogram_{thread_id}", float(thread_id))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Should have existing metrics plus half of num_threads new ones
        expected_new_counters = num_threads // 2
        expected_new_histograms = num_threads // 2
        
        assert len(otel_provider._counters) == 1 + expected_new_counters
        assert len(otel_provider._histograms) == 1 + expected_new_histograms


class TestOTelTelemetryProviderTracing:
    """Tests for tracing functionality."""

    def test_custom_trace_creates_span(self, otel_provider):
        """Test that custom_trace creates a span."""
        span = otel_provider.custom_trace("test_span")
        
        assert span is not None
        assert hasattr(span, "get_span_context")

    def test_custom_trace_with_attributes(self, otel_provider):
        """Test that custom_trace works with attributes."""
        attributes = {"key": "value", "operation": "test"}
        
        span = otel_provider.custom_trace("test_span", attributes=attributes)
        
        assert span is not None

    def test_fastapi_middleware(self, otel_provider):
        """Test that fastapi_middleware can be called."""
        mock_app = MagicMock()
        
        # Should not raise an exception
        otel_provider.fastapi_middleware(mock_app)


class TestOTelTelemetryProviderEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_record_count_with_zero(self, otel_provider):
        """Test that record_count works with zero value."""
        otel_provider.record_count("zero_counter", 0.0)
        
        assert "zero_counter" in otel_provider._counters

    def test_record_count_with_large_value(self, otel_provider):
        """Test that record_count works with large values."""
        otel_provider.record_count("large_counter", 1_000_000.0)
        
        assert "large_counter" in otel_provider._counters

    def test_record_histogram_with_negative_value(self, otel_provider):
        """Test that record_histogram works with negative values."""
        otel_provider.record_histogram("negative_histogram", -10.0)
        
        assert "negative_histogram" in otel_provider._histograms

    def test_record_up_down_counter_with_negative_value(self, otel_provider):
        """Test that record_up_down_counter works with negative values."""
        otel_provider.record_up_down_counter("negative_updown", -5.0)
        
        assert "negative_updown" in otel_provider._up_down_counters

    def test_metric_names_with_special_characters(self, otel_provider):
        """Test that metric names with dots and underscores work."""
        otel_provider.record_count("test.counter_name-special", 1.0)
        otel_provider.record_histogram("test.histogram_name-special", 10.0)
        
        assert "test.counter_name-special" in otel_provider._counters
        assert "test.histogram_name-special" in otel_provider._histograms

    def test_empty_attributes_dict(self, otel_provider):
        """Test that empty attributes dict is handled correctly."""
        otel_provider.record_count("test_counter", 1.0, attributes={})
        
        assert "test_counter" in otel_provider._counters

    def test_none_attributes(self, otel_provider):
        """Test that None attributes are handled correctly."""
        otel_provider.record_count("test_counter", 1.0, attributes=None)
        
        assert "test_counter" in otel_provider._counters

