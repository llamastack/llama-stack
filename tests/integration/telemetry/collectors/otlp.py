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

from .base import BaseTelemetryCollector, MetricStub, SpanStub, attributes_to_dict


class OtlpHttpTestCollector(BaseTelemetryCollector):
    def __init__(self) -> None:
        self._spans: list[SpanStub] = []
        self._metrics: list[MetricStub] = []
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
                    new_spans.append(self._create_span_stub_from_protobuf(span, resource_attrs or None))

        if not new_spans:
            return

        with self._lock:
            self._spans.extend(new_spans)

    def _handle_metrics(self, request: ExportMetricsServiceRequest) -> None:
        new_metrics: list[MetricStub] = []
        for resource_metrics in request.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    metric_stub = self._create_metric_stub_from_protobuf(metric)
                    if metric_stub:
                        new_metrics.append(metric_stub)

        if not new_metrics:
            return

        with self._lock:
            self._metrics.extend(new_metrics)

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:
        with self._lock:
            return tuple(self._spans)

    def _snapshot_metrics(self) -> tuple[MetricStub, ...] | None:
        with self._lock:
            return tuple(self._metrics) if self._metrics else None

    def _clear_impl(self) -> None:
        with self._lock:
            self._spans.clear()
            self._metrics.clear()

    def _create_metric_stub_from_protobuf(self, metric: Any) -> MetricStub | None:
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

                        # Extract value based on metric type
                        if metric_type == "sum":
                            value = data_point.as_int
                        elif metric_type == "gauge":
                            value = data_point.as_double
                        else:  # histogram
                            value = data_point.count

                        # Extract attributes if available
                        attributes = self._extract_attributes_from_data_point(data_point)

                        return MetricStub(
                            name=metric.name,
                            value=value,
                            attributes=attributes if attributes else None,
                        )

        return None

    def _extract_attributes_from_data_point(self, data_point: Any) -> dict[str, Any]:
        """Extract attributes from a protobuf data point."""
        if not hasattr(data_point, "attributes"):
            return {}

        attrs = data_point.attributes
        if not attrs:
            return {}

        return {kv.key: kv.value.string_value or kv.value.int_value or kv.value.double_value for kv in attrs}

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
