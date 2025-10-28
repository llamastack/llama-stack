# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OTLP HTTP telemetry collector used for server-mode tests."""

import gzip
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from .base import BaseTelemetryCollector, SpanStub, attributes_to_dict, events_to_list


class OtlpHttpTestCollector(BaseTelemetryCollector):
    def __init__(self) -> None:
        self._spans: list[SpanStub] = []
        self._metrics: list[Any] = []
        self._lock = threading.Lock()

        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        configured_port = int(os.environ.get("LLAMA_STACK_TEST_COLLECTOR_PORT", "0"))

        self._server = _ThreadingHTTPServer(("127.0.0.1", configured_port), _CollectorHandler)
        self._server.collector = self  # type: ignore[attr-defined]
        port = self._server.server_address[1]
        self.endpoint = f"http://127.0.0.1:{port}"

        self._thread = threading.Thread(target=self._server.serve_forever, name="otel-test-collector", daemon=True)
        self._thread.start()

    def _handle_traces(self, request: ExportTraceServiceRequest) -> None:
        new_spans: list[SpanStub] = []

        for resource_spans in request.resource_spans:
            resource_attrs = attributes_to_dict(resource_spans.resource.attributes)

            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    attributes = attributes_to_dict(span.attributes)
                    events = events_to_list(span.events) if span.events else None
                    trace_id = span.trace_id.hex() if span.trace_id else None
                    span_id = span.span_id.hex() if span.span_id else None
                    new_spans.append(SpanStub(span.name, attributes, resource_attrs or None, events, trace_id, span_id))

        if not new_spans:
            return

        with self._lock:
            self._spans.extend(new_spans)

    def _handle_metrics(self, request: ExportMetricsServiceRequest) -> None:
        new_metrics: list[Any] = []
        for resource_metrics in request.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                new_metrics.extend(scope_metrics.metrics)

        if not new_metrics:
            return

        with self._lock:
            self._metrics.extend(new_metrics)

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:
        with self._lock:
            return tuple(self._spans)

    def _snapshot_metrics(self) -> Any | None:
        with self._lock:
            return list(self._metrics) if self._metrics else None

    def _clear_impl(self) -> None:
        with self._lock:
            self._spans.clear()
            self._metrics.clear()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)


class _CollectorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 Function name `do_POST` should be lowercase
        collector: OtlpHttpTestCollector = self.server.collector  # type: ignore[attr-defined]
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)

        if self.path == "/v1/traces":
            request = ExportTraceServiceRequest()
            request.ParseFromString(body)
            collector._handle_traces(request)
            self._respond_ok()
        elif self.path == "/v1/metrics":
            request = ExportMetricsServiceRequest()
            request.ParseFromString(body)
            collector._handle_metrics(request)
            self._respond_ok()
        else:
            self.send_response(404)
            self.end_headers()

    def _respond_ok(self) -> None:
        self.send_response(200)
        self.end_headers()
