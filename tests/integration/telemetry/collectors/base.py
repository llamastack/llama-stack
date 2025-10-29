# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared helpers for telemetry test collectors."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricStub:
    """Unified metric interface for both in-memory and OTLP collectors."""

    name: str
    value: Any
    attributes: dict[str, Any] | None = None

    def get_value(self) -> Any:
        """Get the metric value."""
        return self.value

    def get_name(self) -> str:
        """Get the metric name."""
        return self.name

    def get_attributes(self) -> dict[str, Any]:
        """Get metric attributes as a dictionary."""
        return self.attributes or {}

    def get_attribute(self, key: str) -> Any:
        """Get a specific attribute value by key."""
        return self.get_attributes().get(key)


@dataclass
class SpanStub:
    """Unified span interface for both in-memory and OTLP collectors."""

    name: str
    attributes: Mapping[str, Any] | None = None
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

    def get_attributes(self) -> dict[str, Any]:
        """Get span attributes as a dictionary.

        Handles different attribute types (mapping, dict, etc.) and returns
        a consistent dictionary format.
        """
        return BaseTelemetryCollector._convert_attributes_to_dict(self.attributes)

    def get_attribute(self, key: str) -> Any:
        """Get a specific attribute value by key."""
        attrs = self.get_attributes()
        return attrs.get(key)

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
        args = self.get_attribute("__args__")
        if not args or not isinstance(args, str):
            return False
        return text in args

    def is_root_span(self) -> bool:
        """Check if this is a root span."""
        return self.get_attribute("__root__") is True

    def is_autotraced(self) -> bool:
        """Check if this span was automatically traced."""
        return self.get_attribute("__autotraced__") is True

    def get_span_type(self) -> str | None:
        """Get the span type (async, sync, async_generator)."""
        return self.get_attribute("__type__")

    def get_class_method(self) -> tuple[str | None, str | None]:
        """Get the class and method names for autotraced spans."""
        return (self.get_attribute("__class__"), self.get_attribute("__method__"))

    def get_location(self) -> str | None:
        """Get the location (library_client, server) for root spans."""
        return self.get_attribute("__location__")


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
        import time

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

    def get_metrics(self) -> tuple[MetricStub, ...] | None:
        return self._snapshot_metrics()

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as a simple name->value dictionary for easy lookup.

        This method works with MetricStub objects for consistent interface
        across both in-memory and OTLP collectors.
        """
        metrics = self._snapshot_metrics()
        if not metrics:
            return {}

        return {metric.get_name(): metric.get_value() for metric in metrics}

    def get_metric_value(self, name: str) -> Any | None:
        """Get a specific metric value by name."""
        return self.get_metrics_dict().get(name)

    def has_metric(self, name: str) -> bool:
        """Check if a metric with the given name exists."""
        return name in self.get_metrics_dict()

    def get_metric_names(self) -> list[str]:
        """Get all available metric names."""
        return list(self.get_metrics_dict().keys())

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
        attributes = BaseTelemetryCollector._convert_attributes_to_dict(span.attributes)

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
        attributes = attributes_to_dict(span.attributes)
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
        value = metric.data.data_points[0].value

        # Extract attributes if available
        attributes = {}
        if hasattr(metric.data.data_points[0], "attributes"):
            attrs = metric.data.data_points[0].attributes
            if attrs is not None and hasattr(attrs, "items"):
                attributes = dict(attrs.items())
            elif attrs is not None and not isinstance(attrs, dict):
                attributes = dict(attrs)

        return MetricStub(
            name=metric.name,
            value=value,
            attributes=attributes if attributes else None,
        )

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
