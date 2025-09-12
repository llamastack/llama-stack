# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.telemetry import MetricEvent, Telemetry
from llama_stack.core.server.metrics import RequestMetricsMiddleware


class TestRequestMetricsMiddleware:
    @pytest.fixture
    def mock_telemetry(self):
        telemetry = AsyncMock(spec=Telemetry)
        return telemetry

    @pytest.fixture
    def mock_app(self):
        app = AsyncMock()
        return app

    @pytest.fixture
    def middleware(self, mock_app, mock_telemetry):
        return RequestMetricsMiddleware(mock_app, mock_telemetry)

    def test_extract_api_from_path(self, middleware):
        """Test API extraction from various paths."""
        test_cases = [
            ("/v1/inference/chat/completions", "inference"),
            ("/v1/models/list", "models"),
            ("/v1/providers", "providers"),
            ("/", "unknown"),
            ("", "unknown"),
        ]

        for path, expected_api in test_cases:
            assert middleware._extract_api_from_path(path) == expected_api

    def test_is_excluded_path(self, middleware):
        """Test path exclusion logic."""
        excluded_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/static/css/style.css",
        ]

        non_excluded_paths = [
            "/v1/inference/chat/completions",
            "/v1/models/list",
            "/health",
        ]

        for path in excluded_paths:
            assert middleware._is_excluded_path(path)

        for path in non_excluded_paths:
            assert not middleware._is_excluded_path(path)

    async def test_middleware_skips_excluded_paths(self, middleware, mock_app):
        """Test that middleware skips metrics for excluded paths."""
        scope = {
            "type": "http",
            "path": "/docs",
            "method": "GET",
        }

        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Should call the app directly without tracking metrics
        mock_app.assert_called_once_with(scope, receive, send)
        # Should not log any metrics
        middleware.telemetry.log_event.assert_not_called()

    async def test_middleware_tracks_metrics(self, middleware, mock_telemetry):
        """Test that middleware tracks metrics for valid requests."""
        scope = {
            "type": "http",
            "path": "/v1/inference/chat/completions",
            "method": "POST",
        }

        receive = AsyncMock()
        send_called = False

        async def mock_send(message):
            nonlocal send_called
            send_called = True
            if message["type"] == "http.response.start":
                message["status"] = 200

        # Mock the app to return successfully
        async def mock_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware.app = mock_app

        await middleware(scope, receive, mock_send)

        # Wait for async metric logging
        await asyncio.sleep(0.1)

        # Should have logged metrics
        assert mock_telemetry.log_event.call_count >= 2

        # Check that the right metrics were logged
        call_args = [call.args[0] for call in mock_telemetry.log_event.call_args_list]

        # Should have request count metric
        request_count_metric = next(
            (
                call
                for call in call_args
                if isinstance(call, MetricEvent) and call.metric == "llama_stack_requests_total"
            ),
            None,
        )
        assert request_count_metric is not None
        assert request_count_metric.value == 1
        assert request_count_metric.attributes["api"] == "inference"
        assert request_count_metric.attributes["status"] == "200"

        # Should have duration metric
        duration_metric = next(
            (
                call
                for call in call_args
                if isinstance(call, MetricEvent) and call.metric == "llama_stack_request_duration_seconds"
            ),
            None,
        )
        assert duration_metric is not None
        assert duration_metric.attributes["api"] == "inference"
        assert duration_metric.attributes["status"] == "200"

    async def test_middleware_handles_errors(self, middleware, mock_telemetry):
        """Test that middleware tracks metrics even when errors occur."""
        scope = {
            "type": "http",
            "path": "/v1/inference/chat/completions",
            "method": "POST",
        }

        receive = AsyncMock()
        send = AsyncMock()

        # Mock the app to raise an exception
        async def mock_app(scope, receive, send):
            raise ValueError("Test error")

        middleware.app = mock_app

        with pytest.raises(ValueError):
            await middleware(scope, receive, send)

        # Wait for async metric logging
        await asyncio.sleep(0.1)

        # Should have logged metrics with error status
        assert mock_telemetry.log_event.call_count >= 2

        # Check that error metrics were logged
        call_args = [call.args[0] for call in mock_telemetry.log_event.call_args_list]

        request_count_metric = next(
            (
                call
                for call in call_args
                if isinstance(call, MetricEvent) and call.metric == "llama_stack_requests_total"
            ),
            None,
        )
        assert request_count_metric is not None
        assert request_count_metric.attributes["status"] == "500"

    async def test_concurrent_requests_tracking(self, middleware, mock_telemetry):
        """Test that concurrent requests are tracked correctly."""
        scope = {
            "type": "http",
            "path": "/v1/inference/chat/completions",
            "method": "POST",
        }

        receive = AsyncMock()
        send = AsyncMock()

        # Mock the app to simulate a slow request
        async def mock_app(scope, receive, send):
            await asyncio.sleep(1)  # Simulate processing time
            await send({"type": "http.response.start", "status": 200})

        middleware.app = mock_app

        # Start multiple concurrent requests
        tasks = []
        for _ in range(3):
            task = asyncio.create_task(middleware(scope, receive, send))
            tasks.append(task)

        # Wait for all requests to complete
        await asyncio.gather(*tasks)

        # Wait for async metric logging
        await asyncio.sleep(0.2)

        # Should have logged metrics for all requests
        assert mock_telemetry.log_event.call_count >= 6  # 2 metrics per request * 3 requests

        # Check concurrent requests metric
        call_args = [call.args[0] for call in mock_telemetry.log_event.call_args_list]
        concurrent_metrics = [
            call
            for call in call_args
            if isinstance(call, MetricEvent) and call.metric == "llama_stack_concurrent_requests"
        ]

        assert len(concurrent_metrics) >= 3
        # The concurrent count should have been > 0 during the concurrent requests
        max_concurrent = max(m.value for m in concurrent_metrics)
        assert max_concurrent > 0

    async def test_middleware_without_telemetry(self):
        """Test that middleware works without telemetry configured."""
        mock_app = AsyncMock()
        middleware = RequestMetricsMiddleware(mock_app, telemetry=None)

        scope = {
            "type": "http",
            "path": "/v1/inference/chat/completions",
            "method": "POST",
        }

        receive = AsyncMock()
        send = AsyncMock()

        async def mock_app_impl(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})

        middleware.app = mock_app_impl

        # Should not raise any exceptions
        await middleware(scope, receive, send)

        # Should not try to log metrics
        # (no telemetry to call, so this is implicit)

    async def test_non_http_requests_ignored(self, middleware, mock_telemetry):
        """Test that non-HTTP requests are ignored."""
        scope = {
            "type": "lifespan",
            "path": "/",
        }

        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Should call the app directly without tracking metrics
        middleware.app.assert_called_once_with(scope, receive, send)
        # Should not log any metrics
        mock_telemetry.log_event.assert_not_called()
