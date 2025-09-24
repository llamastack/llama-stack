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
    """Tests for agent histogram metrics"""

    @pytest.fixture
    def config(self):
        return TelemetryConfig(service_name="test-service", sinks=[])

    @pytest.fixture
    def adapter(self, config):
        adapter = TelemetryAdapter(config, {})
        adapter.meter = Mock()  # skip otel setup
        return adapter

    def test_histogram_creation(self, adapter):
        mock_hist = Mock()
        adapter.meter.create_histogram.return_value = mock_hist

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        result = adapter._get_or_create_histogram("test_histogram", "s")

        assert result == mock_hist
        adapter.meter.create_histogram.assert_called_once_with(
            name="test_histogram",
            unit="s",
            description="test histogram",
        )
        assert _GLOBAL_STORAGE["histograms"]["test_histogram"] == mock_hist

    def test_histogram_reuse(self, adapter):
        mock_hist = Mock()
        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {"existing_histogram": mock_hist}

        result = adapter._get_or_create_histogram("existing_histogram", "ms")

        assert result == mock_hist
        adapter.meter.create_histogram.assert_not_called()

    def test_workflow_duration_histogram(self, adapter):
        mock_hist = Mock()
        adapter.meter.create_histogram.return_value = mock_hist

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="llama_stack_agent_workflow_duration_seconds",
            value=15.7,
            timestamp=1234567890.0,
            unit="s",
            attributes={"agent_id": "test-agent"},
            metric_type=MetricType.HISTOGRAM,
        )

        adapter._log_metric(event)

        adapter.meter.create_histogram.assert_called_once_with(
            name="llama_stack_agent_workflow_duration_seconds",
            unit="s",
            description="llama stack agent workflow duration seconds",
        )
        mock_hist.record.assert_called_once_with(15.7, attributes={"agent_id": "test-agent"})

    def test_duration_buckets_configured_via_views(self, adapter):
        mock_hist = Mock()
        adapter.meter.create_histogram.return_value = mock_hist

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="custom_duration_seconds",
            value=5.2,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )

        adapter._log_metric(event)

        # buckets configured via otel views, not passed to create_histogram
        mock_hist.record.assert_called_once_with(5.2, attributes={})

    def test_non_duration_uses_counter(self, adapter):
        mock_counter = Mock()
        adapter.meter.create_counter.return_value = mock_counter

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["counters"] = {}

        event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="llama_stack_agent_workflows_total",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={"agent_id": "test-agent", "status": "completed"},
        )

        adapter._log_metric(event)

        adapter.meter.create_counter.assert_called_once()
        adapter.meter.create_histogram.assert_not_called()
        mock_counter.add.assert_called_once_with(1, attributes={"agent_id": "test-agent", "status": "completed"})

    def test_no_meter_doesnt_crash(self, adapter):
        adapter.meter = None

        event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
        )

        adapter._log_metric(event)  # shouldn't crash

    def test_histogram_vs_counter_by_type(self, adapter):
        mock_hist = Mock()
        mock_counter = Mock()
        adapter.meter.create_histogram.return_value = mock_hist
        adapter.meter.create_counter.return_value = mock_counter

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}
        _GLOBAL_STORAGE["counters"] = {}

        # histogram metric
        hist_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="workflow_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )
        adapter._log_metric(hist_event)
        mock_hist.record.assert_called()

        # counter metric (default type)
        counter_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="workflow_total",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
        )
        adapter._log_metric(counter_event)
        mock_counter.add.assert_called()

    def test_storage_separation(self, adapter):
        mock_hist = Mock()
        mock_counter = Mock()
        adapter.meter.create_histogram.return_value = mock_hist
        adapter.meter.create_counter.return_value = mock_counter

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}
        _GLOBAL_STORAGE["counters"] = {}

        # create both types
        hist_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_duration_seconds",
            value=1.0,
            timestamp=1234567890.0,
            unit="s",
            attributes={},
            metric_type=MetricType.HISTOGRAM,
        )
        counter_event = MetricEvent(
            trace_id="123",
            span_id="456",
            metric="test_counter",
            value=1,
            timestamp=1234567890.0,
            unit="1",
            attributes={},
        )

        adapter._log_metric(hist_event)
        adapter._log_metric(counter_event)

        # check they're stored separately
        assert "test_duration_seconds" in _GLOBAL_STORAGE["histograms"]
        assert "test_counter" in _GLOBAL_STORAGE["counters"]
        assert "test_duration_seconds" not in _GLOBAL_STORAGE["counters"]
        assert "test_counter" not in _GLOBAL_STORAGE["histograms"]

    def test_histogram_uses_views_for_buckets(self, adapter):
        mock_hist = Mock()
        adapter.meter.create_histogram.return_value = mock_hist

        from llama_stack.providers.inline.telemetry.meta_reference.telemetry import _GLOBAL_STORAGE

        _GLOBAL_STORAGE["histograms"] = {}

        result = adapter._get_or_create_histogram("test_histogram", "s")

        # buckets come from otel views, not create_histogram params
        adapter.meter.create_histogram.assert_called_once_with(
            name="test_histogram",
            unit="s",
            description="test histogram",
        )
        assert result == mock_hist
