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
        attrs = self.attributes
        if attrs is None:
            return {}

        # Handle mapping-like objects (e.g., mappingproxy)
        try:
            return dict(attrs.items())  # type: ignore[attr-defined]
        except AttributeError:
            try:
                return dict(attrs)
            except TypeError:
                return dict(attrs) if attrs else {}

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

    def get_metrics(self) -> Any | None:
        return self._snapshot_metrics()

    def clear(self) -> None:
        self._clear_impl()

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _snapshot_metrics(self) -> Any | None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _clear_impl(self) -> None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def shutdown(self) -> None:
        """Optional hook for subclasses with background workers."""
