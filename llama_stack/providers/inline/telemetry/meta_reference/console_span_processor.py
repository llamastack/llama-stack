# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import UTC, datetime

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace.status import StatusCode

from llama_stack.log import get_logger

logger = get_logger(name="console_span_processor", category="telemetry")


class ConsoleSpanProcessor(SpanProcessor):
    def __init__(self, print_attributes: bool = False):
        self.print_attributes = print_attributes

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        if span.attributes and span.attributes.get("__autotraced__"):
            return

        timestamp = datetime.fromtimestamp(span.start_time / 1e9, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"[dim]{timestamp}[/dim] [bold magenta][START][/bold magenta] [dim]{span.name}[/dim]")

    def on_end(self, span: ReadableSpan) -> None:
        timestamp = datetime.fromtimestamp(span.end_time / 1e9, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        span_context = f"[dim]{timestamp}[/dim] [bold magenta][END][/bold magenta] [dim]{span.name}[/dim]"
        if span.status.status_code == StatusCode.ERROR:
            span_context += " [bold red][ERROR][/bold red]"
        elif span.status.status_code != StatusCode.UNSET:
            span_context += f" [{span.status.status_code}]"
        duration_ms = (span.end_time - span.start_time) / 1e6
        span_context += f" ({duration_ms:.2f}ms)"
        logger.info(span_context)

        if self.print_attributes and span.attributes:
            for key, value in span.attributes.items():
                if key.startswith("__"):
                    continue
                str_value = str(value)
                if len(str_value) > 1000:
                    str_value = str_value[:997] + "..."
                logger.info(f"    [dim]{key}[/dim]: {str_value}")

        for event in span.events:
            event_time = datetime.fromtimestamp(event.timestamp / 1e9, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
            severity = event.attributes.get("severity", "info")
            message = event.attributes.get("message", event.name)
            if isinstance(message, dict) or isinstance(message, list):
                message = json.dumps(message, indent=2)
            severity_color = {
                "error": "red",
                "warn": "yellow",
                "info": "white",
                "debug": "dim",
            }.get(severity, "white")
            logger.info(f" {event_time} [bold {severity_color}][{severity.upper()}][/bold {severity_color}] {message}")
            if event.attributes:
                for key, value in event.attributes.items():
                    if key.startswith("__") or key in ["message", "severity"]:
                        continue
                    logger.info(f"[dim]{key}[/dim]: {value}")

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: float | None = None) -> bool:
        """Force flush any pending spans."""
        return True
