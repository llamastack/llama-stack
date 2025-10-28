# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared helpers for telemetry test collectors."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class SpanStub:
    name: str
    attributes: dict[str, Any]
    resource_attributes: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None
    trace_id: str | None = None
    span_id: str | None = None


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
    def get_spans(
        self,
        expected_count: int | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> tuple[Any, ...]:
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

    def _snapshot_spans(self) -> tuple[Any, ...]:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _snapshot_metrics(self) -> Any | None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _clear_impl(self) -> None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def shutdown(self) -> None:
        """Optional hook for subclasses with background workers."""
