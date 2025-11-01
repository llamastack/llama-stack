# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared helpers for telemetry test collectors."""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricStub:
    """Unified metric interface for both in-memory and OTLP collectors."""

    name: str
    value: Any
    attributes: dict[str, Any] | None = None


@dataclass
class SpanStub:
    """Unified span interface for both in-memory and OTLP collectors."""

    name: str
    attributes: dict[str, Any] | None = None
    resource_attributes: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None
    trace_id: str | None = None
    span_id: str | None = None

    @property
    def context(self):
        """Provide context-like interface for trace_id compatibility."""
        if self.trace_id is None:
            return None
        return type("Context", (), {"trace_id": int(self.trace_id, 16)})()

    def get_trace_id(self) -> str | None:
        """Get trace ID in hex format.

        Tries context.trace_id first, then falls back to direct trace_id.
        """
        context = getattr(self, "context", None)
        if context and getattr(context, "trace_id", None) is not None:
            return f"{context.trace_id:032x}"
        return getattr(self, "trace_id", None)

    def has_message(self, text: str) -> bool:
        """Check if span contains a specific message in its args."""
        if self.attributes is None:
            return False
        args = self.attributes.get("__args__")
        if not args or not isinstance(args, str):
            return False
        return text in args

    def is_root_span(self) -> bool:
        """Check if this is a root span."""
        if self.attributes is None:
            return False
        return self.attributes.get("__root__") is True

    def is_autotraced(self) -> bool:
        """Check if this span was automatically traced."""
        if self.attributes is None:
            return False
        return self.attributes.get("__autotraced__") is True

    def get_span_type(self) -> str | None:
        """Get the span type (async, sync, async_generator)."""
        if self.attributes is None:
            return None
        return self.attributes.get("__type__")

    def get_class_method(self) -> tuple[str | None, str | None]:
        """Get the class and method names for autotraced spans."""
        if self.attributes is None:
            return None, None
        return (self.attributes.get("__class__"), self.attributes.get("__method__"))

    def get_location(self) -> str | None:
        """Get the location (library_client, server) for root spans."""
        if self.attributes is None:
            return None
        return self.attributes.get("__location__")


def _value_to_python(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "bytes_value":
        return value.bytes_value
    if kind == "array_value":
        return [_value_to_python(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _value_to_python(kv.value) for kv in value.kvlist_value.values}
    return None


def attributes_to_dict(key_values: Iterable[Any]) -> dict[str, Any]:
    return {key_value.key: _value_to_python(key_value.value) for key_value in key_values}


def events_to_list(events: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": event.name,
            "timestamp": event.time_unix_nano,
            "attributes": attributes_to_dict(event.attributes),
        }
        for event in events
    ]


class BaseTelemetryCollector:
    """Base class for telemetry collectors that ensures consistent return types.

    All collectors must return SpanStub objects to ensure test compatibility
    across both library-client and server modes.
    """

    def get_spans(
        self,
        expected_count: int | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> tuple[SpanStub, ...]:
        deadline = time.time() + timeout
        min_count = expected_count if expected_count is not None else 1
        last_len: int | None = None
        stable_iterations = 0

        while True:
            spans = tuple(self._snapshot_spans())

            if len(spans) >= min_count:
                if expected_count is not None and len(spans) >= expected_count:
                    return spans

                if last_len == len(spans):
                    stable_iterations += 1
                    if stable_iterations >= 2:
                        return spans
                else:
                    stable_iterations = 1
            else:
                stable_iterations = 0

            if time.time() >= deadline:
                return spans

            last_len = len(spans)
            time.sleep(poll_interval)

    def get_metrics(
        self,
        expected_count: int | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
        expect_model_id: str | None = None,
    ) -> dict[str, MetricStub]:
        """Get metrics with polling until metrics are available or timeout is reached."""

        # metrics need to be collected since get requests delete stored metrics
        deadline = time.time() + timeout
        min_count = expected_count if expected_count is not None else 1
        accumulated_metrics = {}
        count_metrics_with_model_id = 0

        while time.time() < deadline:
            current_metrics = self._snapshot_metrics()
            if current_metrics:
                for metric in current_metrics:
                    metric_name = metric.name
                    if metric_name not in accumulated_metrics:
                        accumulated_metrics[metric_name] = metric
                        if (
                            expect_model_id
                            and metric.attributes
                            and metric.attributes.get("model_id") == expect_model_id
                        ):
                            count_metrics_with_model_id += 1
                    else:
                        accumulated_metrics[metric_name] = metric

            # Check if we have enough metrics
            if len(accumulated_metrics) >= min_count:
                if not expect_model_id:
                    return accumulated_metrics
                if count_metrics_with_model_id >= min_count:
                    return accumulated_metrics

            time.sleep(poll_interval)

        return accumulated_metrics

    @staticmethod
    def _convert_attributes_to_dict(attrs: Any) -> dict[str, Any]:
        """Convert various attribute types to a consistent dictionary format.

        Handles mappingproxy, dict, and other attribute types.
        """
        if attrs is None:
            return {}

        try:
            return dict(attrs.items())  # type: ignore[attr-defined]
        except AttributeError:
            try:
                return dict(attrs)
            except TypeError:
                return dict(attrs) if attrs else {}

    @staticmethod
    def _extract_trace_span_ids(span: Any) -> tuple[str | None, str | None]:
        """Extract trace_id and span_id from OpenTelemetry span object.

        Handles both context-based and direct attribute access.
        """
        trace_id = None
        span_id = None

        context = getattr(span, "context", None)
        if context:
            trace_id = f"{context.trace_id:032x}"
            span_id = f"{context.span_id:016x}"
        else:
            trace_id = getattr(span, "trace_id", None)
            span_id = getattr(span, "span_id", None)

        return trace_id, span_id

    @staticmethod
    def _create_span_stub_from_opentelemetry(span: Any) -> SpanStub:
        """Create SpanStub from OpenTelemetry span object.

        This helper reduces code duplication between collectors.
        """
        trace_id, span_id = BaseTelemetryCollector._extract_trace_span_ids(span)
        attributes = BaseTelemetryCollector._convert_attributes_to_dict(span.attributes) or {}

        return SpanStub(
            name=span.name,
            attributes=attributes,
            trace_id=trace_id,
            span_id=span_id,
        )

    @staticmethod
    def _create_span_stub_from_protobuf(span: Any, resource_attrs: dict[str, Any] | None = None) -> SpanStub:
        """Create SpanStub from protobuf span object.

        This helper handles the different structure of protobuf spans.
        """
        attributes = attributes_to_dict(span.attributes) or {}
        events = events_to_list(span.events) if span.events else None
        trace_id = span.trace_id.hex() if span.trace_id else None
        span_id = span.span_id.hex() if span.span_id else None

        return SpanStub(
            name=span.name,
            attributes=attributes,
            resource_attributes=resource_attrs,
            events=events,
            trace_id=trace_id,
            span_id=span_id,
        )

    @staticmethod
    def _extract_metric_from_opentelemetry(metric: Any) -> MetricStub | None:
        """Extract MetricStub from OpenTelemetry metric object.

        This helper reduces code duplication between collectors.
        """
        if not (hasattr(metric, "name") and hasattr(metric, "data") and hasattr(metric.data, "data_points")):
            return None

        if not (metric.data.data_points and len(metric.data.data_points) > 0):
            return None

        # Get the value from the first data point
        data_point = metric.data.data_points[0]

        # Handle different metric types
        if hasattr(data_point, "value"):
            # Counter or Gauge
            value = data_point.value
        elif hasattr(data_point, "sum"):
            # Histogram - use the sum of all recorded values
            value = data_point.sum
        else:
            return None

        # Extract attributes if available
        attributes = {}
        if hasattr(data_point, "attributes"):
            attrs = data_point.attributes
            if attrs is not None and hasattr(attrs, "items"):
                attributes = dict(attrs.items())
            elif attrs is not None and not isinstance(attrs, dict):
                attributes = dict(attrs)

        return MetricStub(
            name=metric.name,
            value=value,
            attributes=attributes or {},
        )

    @staticmethod
    def _create_metric_stub_from_protobuf(metric: Any) -> MetricStub | None:
        """Create MetricStub from protobuf metric object.

        Protobuf metrics have a different structure than OpenTelemetry metrics.
        They can have sum, gauge, or histogram data.
        """
        if not hasattr(metric, "name"):
            return None

        # Try to extract value from different metric types
        for metric_type in ["sum", "gauge", "histogram"]:
            if hasattr(metric, metric_type):
                metric_data = getattr(metric, metric_type)
                if metric_data and hasattr(metric_data, "data_points"):
                    data_points = metric_data.data_points
                    if data_points and len(data_points) > 0:
                        data_point = data_points[0]

                        # Extract attributes first (needed for all metric types)
                        attributes = (
                            attributes_to_dict(data_point.attributes) if hasattr(data_point, "attributes") else {}
                        )

                        # Extract value based on metric type
                        if metric_type == "sum":
                            value = data_point.as_int
                        elif metric_type == "gauge":
                            value = data_point.as_double
                        else:  # histogram
                            value = data_point.sum

                        return MetricStub(
                            name=metric.name,
                            value=value,
                            attributes=attributes,
                        )
        return None

    def clear(self) -> None:
        self._clear_impl()

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _snapshot_metrics(self) -> tuple[MetricStub, ...] | None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _clear_impl(self) -> None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def shutdown(self) -> None:
        """Optional hook for subclasses with background workers."""
