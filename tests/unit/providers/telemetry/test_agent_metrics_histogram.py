# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

import pytest

from llama_stack.apis.telemetry import MetricEvent, MetricType
from llama_stack.providers.inline.telemetry.meta_reference.config import TelemetryConfig
from llama_stack.providers.inline.telemetry.meta_reference.telemetry import TelemetryAdapter


class TestAgentMetricsHistogram:
    """Unit tests for histogram support in telemetry adapter for agent metrics"""

    @pytest.fixture
    def telemetry_config(self):
        """Basic telemetry config for testing"""
        return TelemetryConfig(
            service_name="test-service",
            sinks=[],
        )

    @pytest.fixture
    def telemetry_adapter(self, telemetry_config):
        """TelemetryAdapter with mocked meter"""
        adapter = TelemetryAdapter(telemetry_config, {})
        # Mock the meter to avoid OpenTelemetry setup
        adapter.meter = Mock()
        return adapter

    def test_get_or_create_histogram_new(self, telemetry_adapter):
        """Test creating a new histogram"""
        mock_histogram = Mock()
        telemetry_adapter.meter.create_histogram.return_value = mock_histogram

        # Clear global storage to ensure clean state
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        result = telemetry_adapter._get_or_create_histogram("test_histogram", "s", [0.1, 0.5, 1.0, 5.0, 10.0])

        assert result == mock_histogram
        telemetry_adapter.meter.create_histogram.assert_called_once_with(
            name="test_histogram",
            unit="s",
            description="Histogram for test_histogram",
        )
        assert _GLOBAL_STORAGE["histograms"]["test_histogram"] == mock_histogram

    def test_get_or_create_histogram_existing(self, telemetry_adapter):
        """Test retrieving an existing histogram"""
        mock_histogram = Mock()

        # Pre-populate global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {"existing_histogram": mock_histogram}

        result = telemetry_adapter._get_or_create_histogram("existing_histogram", "ms")

        assert result == mock_histogram
        # Should not create a new histogram
        telemetry_adapter.meter.create_histogram.assert_not_called()

    def test_log_metric_duration_histogram(self, telemetry_adapter):
        """Test logging duration metrics creates histogram"""
        mock_histogram = Mock()
        telemetry_adapter.meter.create_histogram.return_value = mock_histogram

        # Clear global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        metric_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="llama_stack_agent_workflow_duration_seconds",
            value=15.7,
            timestamp=1234567890.0,
            unit="s",
            attributes={"agent_id": "test-agent"},
            metric_type=MetricType.HISTOGRAM,
        )

        telemetry_adapter._log_metric(metric_event)

        # Verify histogram was created and recorded
        telemetry_adapter.meter.create_histogram.assert_called_once_with(
            name="llama_stack_agent_workflow_duration_seconds",
            unit="s",
            description="Histogram for llama_stack_agent_workflow_duration_seconds",
        )
        mock_histogram.record.assert_called_once_with(15.7, attributes={"agent_id": "test-agent"})

    def test_log_metric_duration_histogram_default_buckets(self, telemetry_adapter):
        """Test that duration metrics use default buckets"""
        mock_histogram = Mock()
        telemetry_adapter.meter.create_histogram.return_value = mock_histogram

        # Clear global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        metric_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="custom_duration_seconds",
            value=5.2,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )

        telemetry_adapter._log_metric(metric_event)

        # Verify histogram was created (buckets are not passed to create_histogram in OpenTelemetry)
        mock_histogram.record.assert_called_once_with(5.2, attributes={})

    def test_log_metric_non_duration_counter(self, telemetry_adapter):
        """Test that non-duration metrics still use counters"""
        mock_counter = Mock()
        telemetry_adapter.meter.create_counter.return_value = mock_counter

        # Clear global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["counters"] = {}

        metric_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="llama_stack_agent_workflows_total",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={"agent_id": "test-agent", "status": "completed"},
        )

        telemetry_adapter._log_metric(metric_event)

        # Verify counter was used, not histogram
        telemetry_adapter.meter.create_counter.assert_called_once()
        telemetry_adapter.meter.create_histogram.assert_not_called()
        mock_counter.add.assert_called_once_with(1, attributes={"agent_id": "test-agent", "status": "completed"})

    def test_log_metric_no_meter(self, telemetry_adapter):
        """Test metric logging when meter is None"""
        telemetry_adapter.meter = None

        metric_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
        )

        # Should not raise exception
        telemetry_adapter._log_metric(metric_event)

    def test_histogram_name_detection_patterns(self, telemetry_adapter):
        """Test various duration metric name patterns"""
        mock_histogram = Mock()
        telemetry_adapter.meter.create_histogram.return_value = mock_histogram

        # Clear global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        duration_metrics = [
            "workflow_duration_seconds",
            "request_duration_seconds",
            "processing_duration_seconds",
            "llama_stack_agent_workflow_duration_seconds",
        ]

        for metric_name in duration_metrics:
            _GLOBAL_STORAGE["histograms"] = {}  # Reset for each test

            metric_event = MetricEvent(
                trace_id="123",
                span_id="456",
                metric=metric_name,
                value=1.0,
                timestamp=1234567890.0,
                unit="s",
                attributes={},
                metric_type=MetricType.HISTOGRAM,
            )

            telemetry_adapter._log_metric(metric_event)
            mock_histogram.record.assert_called()

        # Reset call count for negative test
        mock_histogram.record.reset_mock()
        telemetry_adapter.meter.create_histogram.reset_mock()

        # Test non-duration metric
        non_duration_metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="workflow_total",  # No "_duration_seconds" suffix
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
        )

        telemetry_adapter._log_metric(non_duration_metric)

        # Should not create histogram for non-duration metric
        telemetry_adapter.meter.create_histogram.assert_not_called()
        mock_histogram.record.assert_not_called()

    def test_histogram_global_storage_isolation(self, telemetry_adapter):
        """Test that histogram storage doesn't interfere with counters"""
        mock_histogram = Mock()
        mock_counter = Mock()

        telemetry_adapter.meter.create_histogram.return_value = mock_histogram
        telemetry_adapter.meter.create_counter.return_value = mock_counter

        # Clear global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}
        _GLOBAL_STORAGE["counters"] = {}

        # Create histogram
        duration_metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )
        telemetry_adapter._log_metric(duration_metric)

        # Create counter
        counter_metric = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_counter",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
        )
        telemetry_adapter._log_metric(counter_metric)

        # Verify both were created and stored separately
        assert "test_duration_seconds" in _GLOBAL_STORAGE["histograms"]
        assert "test_counter" in _GLOBAL_STORAGE["counters"]
        assert "test_duration_seconds" not in _GLOBAL_STORAGE["counters"]
        assert "test_counter" not in _GLOBAL_STORAGE["histograms"]

    def test_histogram_buckets_parameter_ignored(self, telemetry_adapter):
        """Test that buckets parameter doesn't affect histogram creation (OpenTelemetry handles buckets internally)"""
        mock_histogram = Mock()
        telemetry_adapter.meter.create_histogram.return_value = mock_histogram

        # Clear global storage
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        # Call with buckets parameter
        result = telemetry_adapter._get_or_create_histogram(
            "test_histogram", "s", buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        )

        # Buckets are not passed to OpenTelemetry create_histogram
        telemetry_adapter.meter.create_histogram.assert_called_once_with(
            name="test_histogram",
            unit="s",
            description="Histogram for test_histogram",
        )
        assert result == mock_histogram
