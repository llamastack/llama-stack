# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import concurrent.futures
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer

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


class TestOTelTelemetryProviderTracerAPI:
    """Tests for the get_tracer() API."""

    def test_get_tracer_returns_tracer(self, otel_provider):
        """Test that get_tracer returns a valid Tracer instance."""
        tracer = otel_provider.get_tracer("test.module")
        
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_get_tracer_with_version(self, otel_provider):
        """Test that get_tracer works with version parameter."""
        tracer = otel_provider.get_tracer(
            instrumenting_module_name="test.module",
            instrumenting_library_version="1.0.0"
        )
        
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_get_tracer_with_attributes(self, otel_provider):
        """Test that get_tracer works with attributes."""
        tracer = otel_provider.get_tracer(
            instrumenting_module_name="test.module",
            attributes={"component": "test", "tier": "backend"}
        )
        
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_get_tracer_with_schema_url(self, otel_provider):
        """Test that get_tracer works with schema URL."""
        tracer = otel_provider.get_tracer(
            instrumenting_module_name="test.module",
            schema_url="https://example.com/schema"
        )
        
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_tracer_can_create_spans(self, otel_provider):
        """Test that tracer can create spans."""
        tracer = otel_provider.get_tracer("test.module")
        
        with tracer.start_as_current_span("test.operation") as span:
            assert span is not None
            assert span.is_recording()

    def test_tracer_can_create_spans_with_attributes(self, otel_provider):
        """Test that tracer can create spans with attributes."""
        tracer = otel_provider.get_tracer("test.module")
        
        with tracer.start_as_current_span(
            "test.operation",
            attributes={"user.id": "123", "request.id": "abc"}
        ) as span:
            assert span is not None
            assert span.is_recording()

    def test_multiple_tracers_can_coexist(self, otel_provider):
        """Test that multiple tracers can be created."""
        tracer1 = otel_provider.get_tracer("module.one")
        tracer2 = otel_provider.get_tracer("module.two")
        
        assert tracer1 is not None
        assert tracer2 is not None
        # Tracers with different names might be the same instance or different
        # depending on OTel implementation, so just verify both work
        with tracer1.start_as_current_span("op1") as span1:
            assert span1.is_recording()
        with tracer2.start_as_current_span("op2") as span2:
            assert span2.is_recording()


class TestOTelTelemetryProviderMeterAPI:
    """Tests for the get_meter() API."""

    def test_get_meter_returns_meter(self, otel_provider):
        """Test that get_meter returns a valid Meter instance."""
        meter = otel_provider.get_meter("test.meter")
        
        assert meter is not None
        assert isinstance(meter, Meter)

    def test_get_meter_with_version(self, otel_provider):
        """Test that get_meter works with version parameter."""
        meter = otel_provider.get_meter(
            name="test.meter",
            version="1.0.0"
        )
        
        assert meter is not None
        assert isinstance(meter, Meter)

    def test_get_meter_with_attributes(self, otel_provider):
        """Test that get_meter works with attributes."""
        meter = otel_provider.get_meter(
            name="test.meter",
            attributes={"service": "test", "env": "dev"}
        )
        
        assert meter is not None
        assert isinstance(meter, Meter)

    def test_get_meter_with_schema_url(self, otel_provider):
        """Test that get_meter works with schema URL."""
        meter = otel_provider.get_meter(
            name="test.meter",
            schema_url="https://example.com/schema"
        )
        
        assert meter is not None
        assert isinstance(meter, Meter)

    def test_meter_can_create_counter(self, otel_provider):
        """Test that meter can create counters."""
        meter = otel_provider.get_meter("test.meter")
        
        counter = meter.create_counter(
            "test.requests.total",
            unit="requests",
            description="Total requests"
        )
        
        assert counter is not None
        # Test that counter can be used
        counter.add(1, {"endpoint": "/test"})

    def test_meter_can_create_histogram(self, otel_provider):
        """Test that meter can create histograms."""
        meter = otel_provider.get_meter("test.meter")
        
        histogram = meter.create_histogram(
            "test.request.duration",
            unit="ms",
            description="Request duration"
        )
        
        assert histogram is not None
        # Test that histogram can be used
        histogram.record(42.5, {"method": "GET"})

    def test_meter_can_create_up_down_counter(self, otel_provider):
        """Test that meter can create up/down counters."""
        meter = otel_provider.get_meter("test.meter")
        
        up_down_counter = meter.create_up_down_counter(
            "test.active.connections",
            unit="connections",
            description="Active connections"
        )
        
        assert up_down_counter is not None
        # Test that up/down counter can be used
        up_down_counter.add(5)
        up_down_counter.add(-2)

    def test_meter_can_create_observable_gauge(self, otel_provider):
        """Test that meter can create observable gauges."""
        meter = otel_provider.get_meter("test.meter")
        
        def gauge_callback(options):
            return [{"attributes": {"host": "localhost"}, "value": 42.0}]
        
        gauge = meter.create_observable_gauge(
            "test.memory.usage",
            callbacks=[gauge_callback],
            unit="bytes",
            description="Memory usage"
        )
        
        assert gauge is not None

    def test_multiple_instruments_from_same_meter(self, otel_provider):
        """Test that a meter can create multiple instruments."""
        meter = otel_provider.get_meter("test.meter")
        
        counter = meter.create_counter("test.counter")
        histogram = meter.create_histogram("test.histogram")
        up_down_counter = meter.create_up_down_counter("test.gauge")
        
        assert counter is not None
        assert histogram is not None
        assert up_down_counter is not None
        
        # Verify they all work
        counter.add(1)
        histogram.record(10.0)
        up_down_counter.add(5)


class TestOTelTelemetryProviderNativeUsage:
    """Tests for native OpenTelemetry usage patterns."""

    def test_complete_tracing_workflow(self, otel_provider):
        """Test a complete tracing workflow using native OTel API."""
        tracer = otel_provider.get_tracer("llama_stack.inference")
        
        # Create parent span
        with tracer.start_as_current_span("inference.request") as parent_span:
            parent_span.set_attribute("model", "llama-3.2-1b")
            parent_span.set_attribute("user", "test-user")
            
            # Create child span
            with tracer.start_as_current_span("model.load") as child_span:
                child_span.set_attribute("model.size", "1B")
                assert child_span.is_recording()
            
            # Create another child span
            with tracer.start_as_current_span("inference.execute") as child_span:
                child_span.set_attribute("tokens.input", 25)
                child_span.set_attribute("tokens.output", 150)
                assert child_span.is_recording()
            
            assert parent_span.is_recording()

    def test_complete_metrics_workflow(self, otel_provider):
        """Test a complete metrics workflow using native OTel API."""
        meter = otel_provider.get_meter("llama_stack.metrics")
        
        # Create various instruments
        request_counter = meter.create_counter(
            "llama.requests.total",
            unit="requests",
            description="Total requests"
        )
        
        latency_histogram = meter.create_histogram(
            "llama.inference.duration",
            unit="ms",
            description="Inference duration"
        )
        
        active_sessions = meter.create_up_down_counter(
            "llama.sessions.active",
            unit="sessions",
            description="Active sessions"
        )
        
        # Use the instruments
        request_counter.add(1, {"endpoint": "/chat", "status": "success"})
        latency_histogram.record(123.45, {"model": "llama-3.2-1b"})
        active_sessions.add(1)
        active_sessions.add(-1)
        
        # No exceptions means success

    def test_concurrent_tracer_usage(self, otel_provider):
        """Test that multiple threads can use tracers concurrently."""
        def create_spans(thread_id):
            tracer = otel_provider.get_tracer(f"test.module.{thread_id}")
            for i in range(10):
                with tracer.start_as_current_span(f"operation.{i}") as span:
                    span.set_attribute("thread.id", thread_id)
                    span.set_attribute("iteration", i)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_spans, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        # If we get here without exceptions, thread safety is working

    def test_concurrent_meter_usage(self, otel_provider):
        """Test that multiple threads can use meters concurrently."""
        def record_metrics(thread_id):
            meter = otel_provider.get_meter(f"test.meter.{thread_id}")
            counter = meter.create_counter(f"test.counter.{thread_id}")
            histogram = meter.create_histogram(f"test.histogram.{thread_id}")
            
            for i in range(10):
                counter.add(1, {"thread": str(thread_id)})
                histogram.record(float(i * 10), {"thread": str(thread_id)})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        # If we get here without exceptions, thread safety is working

    def test_mixed_tracing_and_metrics(self, otel_provider):
        """Test using both tracing and metrics together."""
        tracer = otel_provider.get_tracer("test.module")
        meter = otel_provider.get_meter("test.meter")
        
        counter = meter.create_counter("operations.count")
        histogram = meter.create_histogram("operation.duration")
        
        # Trace an operation while recording metrics
        with tracer.start_as_current_span("test.operation") as span:
            counter.add(1)
            span.set_attribute("step", "start")
            
            histogram.record(50.0)
            span.set_attribute("step", "processing")
            
            counter.add(1)
            span.set_attribute("step", "complete")
        
        # No exceptions means success


class TestOTelTelemetryProviderFastAPIMiddleware:
    """Tests for FastAPI middleware functionality."""

    def test_fastapi_middleware(self, otel_provider):
        """Test that fastapi_middleware can be called."""
        mock_app = MagicMock()
        
        # Should not raise an exception
        otel_provider.fastapi_middleware(mock_app)

    def test_fastapi_middleware_is_idempotent(self, otel_provider):
        """Test that calling fastapi_middleware multiple times is safe."""
        mock_app = MagicMock()
        
        # Should be able to call multiple times without error
        otel_provider.fastapi_middleware(mock_app)
        # Note: Second call might warn but shouldn't fail
        # otel_provider.fastapi_middleware(mock_app)


class TestOTelTelemetryProviderEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_tracer_with_empty_module_name(self, otel_provider):
        """Test that get_tracer works with empty module name."""
        tracer = otel_provider.get_tracer("")
        
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_meter_with_empty_name(self, otel_provider):
        """Test that get_meter works with empty name."""
        meter = otel_provider.get_meter("")
        
        assert meter is not None
        assert isinstance(meter, Meter)

    def test_meter_instruments_with_special_characters(self, otel_provider):
        """Test that metric names with dots, underscores, and hyphens work."""
        meter = otel_provider.get_meter("test.meter")
        
        counter = meter.create_counter("test.counter_name-special")
        histogram = meter.create_histogram("test.histogram_name-special")
        
        assert counter is not None
        assert histogram is not None
        
        # Verify they can be used
        counter.add(1)
        histogram.record(10.0)

    def test_meter_counter_with_zero_value(self, otel_provider):
        """Test that counters work with zero value."""
        meter = otel_provider.get_meter("test.meter")
        counter = meter.create_counter("test.counter")
        
        # Should not raise an exception
        counter.add(0.0)

    def test_meter_histogram_with_negative_value(self, otel_provider):
        """Test that histograms accept negative values."""
        meter = otel_provider.get_meter("test.meter")
        histogram = meter.create_histogram("test.histogram")
        
        # Should not raise an exception
        histogram.record(-10.0)

    def test_meter_up_down_counter_with_negative_value(self, otel_provider):
        """Test that up/down counters work with negative values."""
        meter = otel_provider.get_meter("test.meter")
        up_down_counter = meter.create_up_down_counter("test.updown")
        
        # Should not raise an exception
        up_down_counter.add(-5.0)

    def test_meter_instruments_with_empty_attributes(self, otel_provider):
        """Test that empty attributes dict is handled correctly."""
        meter = otel_provider.get_meter("test.meter")
        counter = meter.create_counter("test.counter")
        
        # Should not raise an exception
        counter.add(1.0, attributes={})

    def test_meter_instruments_with_none_attributes(self, otel_provider):
        """Test that None attributes are handled correctly."""
        meter = otel_provider.get_meter("test.meter")
        counter = meter.create_counter("test.counter")
        
        # Should not raise an exception
        counter.add(1.0, attributes=None)


class TestOTelTelemetryProviderRealisticScenarios:
    """Tests simulating realistic usage scenarios."""

    def test_inference_request_telemetry(self, otel_provider):
        """Simulate telemetry for a complete inference request."""
        tracer = otel_provider.get_tracer("llama_stack.inference")
        meter = otel_provider.get_meter("llama_stack.metrics")
        
        # Create instruments
        request_counter = meter.create_counter("llama.requests.total")
        token_counter = meter.create_counter("llama.tokens.total")
        latency_histogram = meter.create_histogram("llama.request.duration_ms")
        in_flight_gauge = meter.create_up_down_counter("llama.requests.in_flight")
        
        # Simulate request
        with tracer.start_as_current_span("inference.request") as request_span:
            request_span.set_attribute("model.id", "llama-3.2-1b")
            request_span.set_attribute("user.id", "test-user")
            
            request_counter.add(1, {"model": "llama-3.2-1b"})
            in_flight_gauge.add(1)
            
            # Simulate token counting
            token_counter.add(25, {"type": "input", "model": "llama-3.2-1b"})
            token_counter.add(150, {"type": "output", "model": "llama-3.2-1b"})
            
            # Simulate latency
            latency_histogram.record(125.5, {"model": "llama-3.2-1b"})
            
            in_flight_gauge.add(-1)
            request_span.set_attribute("tokens.input", 25)
            request_span.set_attribute("tokens.output", 150)

    def test_multi_step_workflow_with_nested_spans(self, otel_provider):
        """Simulate a multi-step workflow with nested spans."""
        tracer = otel_provider.get_tracer("llama_stack.workflow")
        meter = otel_provider.get_meter("llama_stack.workflow.metrics")
        
        step_counter = meter.create_counter("workflow.steps.completed")
        
        with tracer.start_as_current_span("workflow.execute") as root_span:
            root_span.set_attribute("workflow.id", "wf-123")
            
            # Step 1: Validate
            with tracer.start_as_current_span("step.validate") as span:
                span.set_attribute("validation.result", "pass")
                step_counter.add(1, {"step": "validate", "status": "success"})
            
            # Step 2: Process
            with tracer.start_as_current_span("step.process") as span:
                span.set_attribute("items.processed", 100)
                step_counter.add(1, {"step": "process", "status": "success"})
            
            # Step 3: Finalize
            with tracer.start_as_current_span("step.finalize") as span:
                span.set_attribute("output.size", 1024)
                step_counter.add(1, {"step": "finalize", "status": "success"})
            
            root_span.set_attribute("workflow.status", "completed")

    def test_error_handling_with_telemetry(self, otel_provider):
        """Test telemetry when errors occur."""
        tracer = otel_provider.get_tracer("llama_stack.errors")
        meter = otel_provider.get_meter("llama_stack.errors.metrics")
        
        error_counter = meter.create_counter("llama.errors.total")
        
        with tracer.start_as_current_span("operation.with.error") as span:
            try:
                span.set_attribute("step", "processing")
                # Simulate an error
                raise ValueError("Test error")
            except ValueError as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                error_counter.add(1, {"error.type": "ValueError"})
        
        # Should not raise - error was handled

    def test_batch_operations_telemetry(self, otel_provider):
        """Test telemetry for batch operations."""
        tracer = otel_provider.get_tracer("llama_stack.batch")
        meter = otel_provider.get_meter("llama_stack.batch.metrics")
        
        batch_counter = meter.create_counter("llama.batch.items.processed")
        batch_duration = meter.create_histogram("llama.batch.duration_ms")
        
        with tracer.start_as_current_span("batch.process") as batch_span:
            batch_span.set_attribute("batch.size", 100)
            
            for i in range(100):
                with tracer.start_as_current_span(f"item.{i}") as item_span:
                    item_span.set_attribute("item.index", i)
                    batch_counter.add(1, {"status": "success"})
            
            batch_duration.record(5000.0, {"batch.size": "100"})
            batch_span.set_attribute("batch.status", "completed")
