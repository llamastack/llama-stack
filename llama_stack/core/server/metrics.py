# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from datetime import UTC, datetime

from llama_stack.apis.telemetry import MetricEvent, Telemetry
from llama_stack.log import get_logger
from llama_stack.providers.utils.telemetry.tracing import get_current_span

logger = get_logger(name=__name__, category="server")


class RequestMetricsMiddleware:
    """
    middleware that tracks request-level metrics including:
    - Request counts by API and status
    - Request duration
    - Concurrent requests

    Metrics are logged to the telemetry system and can be exported to Prometheus
    via OpenTelemetry.
    """

    def __init__(self, app, telemetry: Telemetry | None = None):
        self.app = app
        self.telemetry = telemetry
        self.concurrent_requests = 0
        self._lock = asyncio.Lock()

        # FastAPI built-in paths that should be excluded from metrics
        self.excluded_paths = ("/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static")

    def _extract_api_from_path(self, path: str) -> str:
        """Extract the API name from the request path."""
        # Remove version prefix if present
        if path.startswith("/v1/"):
            path = path[4:]
        # Extract the first path segment as the API name
        segments = path.strip("/").split("/")
        if (
            segments and segments[0]
        ):  # Check that first segment is not empty, this will return the API rather than the action `["datasets", "list"]`
            return segments[0]
        return "unknown"

    def _is_excluded_path(self, path: str) -> bool:
        """Check if the path should be excluded from metrics."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)

    async def _log_request_metrics(self, api: str, status: str, duration: float, concurrent_count: int):
        """Log request metrics to the telemetry system."""
        if not self.telemetry:
            return

        try:
            # Get current span if available
            span = get_current_span()
            trace_id = span.trace_id if span else ""
            span_id = span.span_id if span else ""

            # Log request count (send increment of 1 for each request)
            await self.telemetry.log_event(
                MetricEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    timestamp=datetime.now(UTC),
                    metric="llama_stack_requests_total",
                    value=1,  # Send increment instead of total so the provider handles incrementation
                    unit="requests",
                    attributes={
                        "api": api,
                        "status": status,
                    },
                )
            )

            # Log request duration
            await self.telemetry.log_event(
                MetricEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    timestamp=datetime.now(UTC),
                    metric="llama_stack_request_duration_seconds",
                    value=duration,
                    unit="seconds",
                    attributes={"api": api, "status": status},
                )
            )

            # Log concurrent requests (as a gauge)
            await self.telemetry.log_event(
                MetricEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    timestamp=datetime.now(UTC),
                    metric="llama_stack_concurrent_requests",
                    value=float(concurrent_count),  # Convert to float for gauge
                    unit="requests",
                    attributes={"api": api},
                )
            )

        except ValueError as e:
            logger.warning(f"Failed to log request metrics: {e}")

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        # Skip metrics for excluded paths
        if self._is_excluded_path(path):
            return await self.app(scope, receive, send)

        api = self._extract_api_from_path(path)
        start_time = time.time()
        status = 200

        # Track concurrent requests
        async with self._lock:
            self.concurrent_requests += 1

        # Create a wrapper to capture the response status
        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                nonlocal status
                status = message.get("status", 200)
            await send(message)

        try:
            return await self.app(scope, receive, send_wrapper)

        except Exception:
            # Set status to 500 for any unhandled exception
            status = 500
            raise

        finally:
            duration = time.time() - start_time

            # Capture concurrent count before decrementing
            async with self._lock:
                concurrent_count = self.concurrent_requests
                self.concurrent_requests -= 1

            # Log metrics asynchronously to avoid blocking the response
            asyncio.create_task(self._log_request_metrics(api, str(status), duration, concurrent_count))
