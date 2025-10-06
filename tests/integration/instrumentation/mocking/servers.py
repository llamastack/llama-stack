# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Mock servers for OpenTelemetry E2E testing.

This module provides mock servers for testing telemetry:
- MockOTLPCollector: Receives and stores OTLP telemetry exports
- MockVLLMServer: Simulates vLLM inference API with valid OpenAI responses

These mocks allow E2E testing without external dependencies.
"""

import asyncio
import http.server
import json
import socket
import threading
import time
from collections import defaultdict
from typing import Any

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from pydantic import Field

from .mock_base import MockServerBase


class MockOTLPCollector(MockServerBase):
    """
    Mock OTLP collector HTTP server.

    Receives real OTLP exports from Llama Stack and stores them for verification.
    Runs on localhost:4318 (standard OTLP HTTP port).

    Usage:
        collector = MockOTLPCollector()
        await collector.await_start()
        # ... run tests ...
        print(f"Received {collector.get_trace_count()} traces")
        collector.stop()
    """

    port: int = Field(default=4318, description="Port to run collector on")

    # Non-Pydantic fields (set after initialization)
    traces: list[dict] = Field(default_factory=list, exclude=True)
    metrics: list[dict] = Field(default_factory=list, exclude=True)
    all_http_requests: list[dict] = Field(default_factory=list, exclude=True)  # Track ALL HTTP requests for debugging
    server: Any = Field(default=None, exclude=True)
    server_thread: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        """Initialize after Pydantic validation."""
        self.traces = []
        self.metrics = []
        self.server = None
        self.server_thread = None

    def _create_handler_class(self):
        """Create the HTTP handler class for this collector instance."""
        collector_self = self

        class OTLPHandler(http.server.BaseHTTPRequestHandler):
            """HTTP request handler for OTLP requests."""

            def log_message(self, format, *args):
                """Suppress HTTP server logs."""
                pass

            def do_GET(self):  # noqa: N802
                """Handle GET requests."""
                # No readiness endpoint needed - using await_start() instead
                self.send_response(404)
                self.end_headers()

            def do_POST(self):  # noqa: N802
                """Handle OTLP POST requests."""
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b""

                # Track ALL requests for debugging
                collector_self.all_http_requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "timestamp": time.time(),
                        "body_length": len(body),
                    }
                )

                # Store the export request
                if "/v1/traces" in self.path:
                    collector_self.traces.append(
                        {
                            "body": body,
                            "timestamp": time.time(),
                        }
                    )
                elif "/v1/metrics" in self.path:
                    collector_self.metrics.append(
                        {
                            "body": body,
                            "timestamp": time.time(),
                        }
                    )

                # Always return success (200 OK)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        return OTLPHandler

    async def await_start(self):
        """
        Start the OTLP collector and wait until ready.

        This method is async and can be awaited to ensure the server is ready.
        """
        # Create handler and start the HTTP server
        handler_class = self._create_handler_class()
        self.server = http.server.HTTPServer(("localhost", self.port), handler_class)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        # Wait for server to be listening on the port
        for _ in range(10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("localhost", self.port))
                sock.close()
                if result == 0:
                    # Port is listening
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)

        raise RuntimeError(f"OTLP collector failed to start on port {self.port}")

    def stop(self):
        """Stop the OTLP collector server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def clear(self):
        """Clear all captured telemetry data."""
        self.traces = []
        self.metrics = []

    def get_trace_count(self) -> int:
        """Get number of trace export requests received."""
        return len(self.traces)

    def get_metric_count(self) -> int:
        """Get number of metric export requests received."""
        return len(self.metrics)

    def get_all_traces(self) -> list[dict]:
        """Get all captured trace exports."""
        return self.traces

    def get_all_metrics(self) -> list[dict]:
        """Get all captured metric exports."""
        return self.metrics

    # -----------------------------
    # Trace parsing helpers
    # -----------------------------
    def parse_traces(self) -> dict[str, list[dict]]:
        """
        Parse protobuf trace data and return spans grouped by trace ID.

        Returns:
            Dict mapping trace_id (hex) -> list of span dicts
        """
        trace_id_to_spans: dict[str, list[dict]] = {}

        for export in self.traces:
            request = ExportTraceServiceRequest()
            body = export.get("body", b"")
            try:
                request.ParseFromString(body)
            except Exception as e:
                raise RuntimeError(f"Failed to parse OTLP traces export (len={len(body)}): {e}") from e

            for resource_span in request.resource_spans:
                for scope_span in resource_span.scope_spans:
                    for span in scope_span.spans:
                        # span.trace_id is bytes; convert to hex string
                        trace_id = (
                            span.trace_id.hex() if isinstance(span.trace_id, bytes | bytearray) else str(span.trace_id)
                        )
                        span_entry = {
                            "name": span.name,
                            "span_id": span.span_id.hex()
                            if isinstance(span.span_id, bytes | bytearray)
                            else str(span.span_id),
                            "start_time_unix_nano": int(getattr(span, "start_time_unix_nano", 0)),
                            "end_time_unix_nano": int(getattr(span, "end_time_unix_nano", 0)),
                        }
                        trace_id_to_spans.setdefault(trace_id, []).append(span_entry)

        return trace_id_to_spans

    def get_all_trace_ids(self) -> set[str]:
        """Return set of all trace IDs seen so far."""
        return set(self.parse_traces().keys())

    def get_trace_span_counts(self) -> dict[str, int]:
        """Return span counts per trace ID."""
        grouped = self.parse_traces()
        return {tid: len(spans) for tid, spans in grouped.items()}

    def get_new_trace_ids(self, prior_ids: set[str]) -> set[str]:
        """Return trace IDs that appeared after prior_ids snapshot."""
        return self.get_all_trace_ids() - set(prior_ids)

    def parse_metrics(self) -> dict[str, list[Any]]:
        """
        Parse protobuf metric data and return metrics by name.

        Returns:
            Dict mapping metric names to list of metric data points
        """
        metrics_by_name = defaultdict(list)

        for export in self.metrics:
            # Parse the protobuf body
            request = ExportMetricsServiceRequest()
            body = export.get("body", b"")
            try:
                request.ParseFromString(body)
            except Exception as e:
                raise RuntimeError(f"Failed to parse OTLP metrics export (len={len(body)}): {e}") from e

            # Extract metrics from the request
            for resource_metric in request.resource_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        metric_name = metric.name

                        # Extract data points based on metric type
                        data_points = []
                        if metric.HasField("gauge"):
                            data_points = list(metric.gauge.data_points)
                        elif metric.HasField("sum"):
                            data_points = list(metric.sum.data_points)
                        elif metric.HasField("histogram"):
                            data_points = list(metric.histogram.data_points)
                        elif metric.HasField("summary"):
                            data_points = list(metric.summary.data_points)

                        metrics_by_name[metric_name].extend(data_points)

        return dict(metrics_by_name)

    def get_metric_by_name(self, metric_name: str) -> list[Any]:
        """
        Get all data points for a specific metric by name.

        Args:
            metric_name: The name of the metric to retrieve

        Returns:
            List of data points for the metric, or empty list if not found
        """
        metrics = self.parse_metrics()
        return metrics.get(metric_name, [])

    def has_metric(self, metric_name: str) -> bool:
        """
        Check if a metric with the given name has been captured.

        Args:
            metric_name: The name of the metric to check

        Returns:
            True if the metric exists and has data points, False otherwise
        """
        data_points = self.get_metric_by_name(metric_name)
        return len(data_points) > 0

    def get_all_metric_names(self) -> list[str]:
        """
        Get all unique metric names that have been captured.

        Returns:
            List of metric names
        """
        return list(self.parse_metrics().keys())


class MockVLLMServer(MockServerBase):
    """
    Mock vLLM inference server with OpenAI-compatible API.

    Returns valid OpenAI Python client response objects for:
    - Chat completions (/v1/chat/completions)
    - Text completions (/v1/completions)
    - Model listing (/v1/models)

    Runs on localhost:8000 (standard vLLM port).

    Usage:
        server = MockVLLMServer(models=["my-model"])
        await server.await_start()
        # ... make inference calls ...
        print(f"Handled {server.get_request_count()} requests")
        server.stop()
    """

    port: int = Field(default=8000, description="Port to run server on")
    models: list[str] = Field(
        default_factory=lambda: ["meta-llama/Llama-3.2-1B-Instruct"], description="List of model IDs to serve"
    )

    # Non-Pydantic fields
    requests_received: list[dict] = Field(default_factory=list, exclude=True)
    server: Any = Field(default=None, exclude=True)
    server_thread: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        """Initialize after Pydantic validation."""
        self.requests_received = []
        self.server = None
        self.server_thread = None

    def _create_handler_class(self):
        """Create the HTTP handler class for this vLLM instance."""
        server_self = self

        class VLLMHandler(http.server.BaseHTTPRequestHandler):
            """HTTP request handler for vLLM API."""

            def log_message(self, format, *args):
                """Suppress HTTP server logs."""
                pass

            def log_request(self, code="-", size="-"):
                """Log incoming requests for debugging."""
                print(f"[DEBUG] Mock vLLM received: {self.command} {self.path} -> {code}")

            def do_GET(self):  # noqa: N802
                """Handle GET requests (models list, health check)."""
                # Log GET requests too
                server_self.requests_received.append(
                    {
                        "path": self.path,
                        "method": "GET",
                        "timestamp": time.time(),
                    }
                )

                if self.path == "/v1/models":
                    response = self._create_models_list_response()
                    self._send_json_response(200, response)

                elif self.path == "/health" or self.path == "/v1/health":
                    self._send_json_response(200, {"status": "healthy"})

                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):  # noqa: N802
                """Handle POST requests (chat/text completions)."""
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"

                try:
                    request_data = json.loads(body)
                except Exception:
                    request_data = {}

                # Log the request
                server_self.requests_received.append(
                    {
                        "path": self.path,
                        "body": request_data,
                        "timestamp": time.time(),
                    }
                )

                # Route to appropriate handler
                if "/chat/completions" in self.path:
                    response = self._create_chat_completion_response(request_data)
                    if response is not None:  # None means already sent (streaming)
                        self._send_json_response(200, response)

                elif "/completions" in self.path:
                    response = self._create_text_completion_response(request_data)
                    self._send_json_response(200, response)

                else:
                    self._send_json_response(200, {"status": "ok"})

            # ----------------------------------------------------------------
            # Response Generators
            # **TO MODIFY RESPONSES:** Edit these methods
            # ----------------------------------------------------------------

            def _create_models_list_response(self) -> dict:
                """Create OpenAI models list response with configured models."""
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "meta",
                        }
                        for model_id in server_self.models
                    ],
                }

            def _create_chat_completion_response(self, request_data: dict) -> dict | None:
                """
                Create OpenAI ChatCompletion response.

                Returns a valid response matching openai.types.ChatCompletion.
                Supports both regular and streaming responses.
                Returns None for streaming responses (already sent via SSE).
                """
                # Check if streaming is requested
                is_streaming = request_data.get("stream", False)

                if is_streaming:
                    # Return SSE streaming response
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    # Send streaming chunks
                    chunks = [
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request_data.get("model", "test"),
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        },
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request_data.get("model", "test"),
                            "choices": [{"index": 0, "delta": {"content": "Test "}, "finish_reason": None}],
                        },
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request_data.get("model", "test"),
                            "choices": [{"index": 0, "delta": {"content": "streaming "}, "finish_reason": None}],
                        },
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request_data.get("model", "test"),
                            "choices": [{"index": 0, "delta": {"content": "response"}, "finish_reason": None}],
                        },
                        {
                            "id": "chatcmpl-test",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request_data.get("model", "test"),
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        },
                    ]

                    for chunk in chunks:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return None  # Already sent response

                # Regular response
                return {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", "meta-llama/Llama-3.2-1B-Instruct"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "This is a test response from mock vLLM server.",
                                "tool_calls": None,
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 15,
                        "total_tokens": 40,
                        "completion_tokens_details": None,
                    },
                    "system_fingerprint": None,
                    "service_tier": None,
                }

            def _create_text_completion_response(self, request_data: dict) -> dict:
                """
                Create OpenAI Completion response.

                Returns a valid response matching openai.types.Completion
                """
                return {
                    "id": "cmpl-test123",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", "meta-llama/Llama-3.2-1B-Instruct"),
                    "choices": [
                        {
                            "text": "This is a test completion.",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                        "completion_tokens_details": None,
                    },
                    "system_fingerprint": None,
                }

            def _send_json_response(self, status_code: int, data: dict):
                """Helper to send JSON response."""
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

        return VLLMHandler

    async def await_start(self):
        """
        Start the vLLM server and wait until ready.

        This method is async and can be awaited to ensure the server is ready.
        """
        # Create handler and start the HTTP server
        handler_class = self._create_handler_class()
        self.server = http.server.HTTPServer(("localhost", self.port), handler_class)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        # Wait for server to be listening on the port
        for _ in range(10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("localhost", self.port))
                sock.close()
                if result == 0:
                    # Port is listening
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)

        raise RuntimeError(f"vLLM server failed to start on port {self.port}")

    def stop(self):
        """Stop the vLLM server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def clear(self):
        """Clear request history."""
        self.requests_received = []

    def get_request_count(self) -> int:
        """Get number of requests received."""
        return len(self.requests_received)

    def get_all_requests(self) -> list[dict]:
        """Get all received requests with their bodies."""
        return self.requests_received
