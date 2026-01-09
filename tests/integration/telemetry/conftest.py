# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry test configuration supporting both library and server test modes."""

import os

import pytest

from llama_stack.testing.api_recorder import patch_httpx_for_test_id
from tests.integration.fixtures.common import instantiate_llama_stack_client
from tests.integration.telemetry.collectors import InMemoryTelemetryManager


@pytest.fixture(scope="session")
def telemetry_test_collector():
    """Provide telemetry collector for capturing metrics and traces.

    In server mode, connects to the OTLP HTTP collector started by the integration test script.
    In library_client mode, creates an in-memory collector via manual MeterProvider setup.

    The server must be started with OTEL_EXPORTER_OTLP_ENDPOINT pointing to this collector.
    """
    stack_mode = os.environ.get("LLAMA_STACK_TEST_STACK_CONFIG_TYPE", "library_client")

    if stack_mode == "server":
        # In server mode, the collector is already running (started by scripts/integration-tests.sh).
        # We just need to create a client connection to it.
        # The collector port is set by LLAMA_STACK_TEST_COLLECTOR_PORT.
        import http.client
        import time

        collector_port = int(os.environ.get("LLAMA_STACK_TEST_COLLECTOR_PORT", "4318"))
        endpoint = f"http://127.0.0.1:{collector_port}"

        # Verify the collector is responding
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=1)
                conn.request("GET", "/")
                conn.getresponse()
                conn.close()
                break
            except Exception:
                if attempt == max_attempts - 1:
                    pytest.skip(f"OTLP collector not responding at {endpoint}")
                time.sleep(0.1)

        # Create a simple collector wrapper that connects to the running collector
        # We can't use OtlpHttpTestCollector directly as it would try to start a new server

        class ExistingCollectorWrapper:
            """Wrapper for accessing an already-running OTLP collector."""

            def __init__(self, endpoint: str):
                self.endpoint = endpoint
                # We'll access the collector's data via HTTP requests
                # The actual collector is running in a separate process

            def get_spans(self, **kwargs):
                """Query spans from the running collector via HTTP."""
                import http.client
                import json

                try:
                    conn = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=5)
                    conn.request("GET", "/query/spans")
                    response = conn.getresponse()
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        from tests.integration.telemetry.collectors.base import SpanStub

                        spans = [
                            SpanStub(
                                name=s["name"],
                                attributes=s.get("attributes"),
                                resource_attributes=s.get("resource_attributes"),
                                trace_id=s.get("trace_id"),
                                span_id=s.get("span_id"),
                            )
                            for s in data.get("spans", [])
                        ]
                        return tuple(spans)
                    conn.close()
                except Exception:
                    pass
                return tuple()

            def get_metrics(self, **kwargs):
                """Query metrics from the running collector via HTTP."""
                import http.client
                import json
                import time

                # Support polling similar to base class
                timeout = kwargs.get("timeout", 5.0)
                poll_interval = kwargs.get("poll_interval", 0.05)
                deadline = time.time() + timeout

                while time.time() < deadline:
                    try:
                        conn = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=5)
                        conn.request("GET", "/query/metrics")
                        response = conn.getresponse()
                        if response.status == 200:
                            data = json.loads(response.read().decode())
                            from tests.integration.telemetry.collectors.base import (
                                MetricStub,
                            )

                            metrics_dict = {}
                            for m in data.get("metrics", []):
                                metric = MetricStub(
                                    name=m["name"],
                                    value=m["value"],
                                    attributes=m.get("attributes"),
                                )
                                # Use the same accumulation logic - keep latest/highest value
                                if m["name"] not in metrics_dict or metric.value > metrics_dict[m["name"]].value:
                                    metrics_dict[m["name"]] = metric

                            conn.close()

                            # Check if we have enough metrics
                            expected_count = kwargs.get("expected_count")
                            if expected_count is None or len(metrics_dict) >= expected_count:
                                return metrics_dict

                    except Exception:
                        pass

                    time.sleep(poll_interval)

                # Return whatever we collected even if timeout
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=5)
                    conn.request("GET", "/query/metrics")
                    response = conn.getresponse()
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        from tests.integration.telemetry.collectors.base import (
                            MetricStub,
                        )

                        metrics_dict = {}
                        for m in data.get("metrics", []):
                            metric = MetricStub(
                                name=m["name"],
                                value=m["value"],
                                attributes=m.get("attributes"),
                            )
                            if m["name"] not in metrics_dict or metric.value > metrics_dict[m["name"]].value:
                                metrics_dict[m["name"]] = metric
                        conn.close()
                        return metrics_dict
                except Exception:
                    pass

                return {}

            def clear(self):
                """Clear the external collector via HTTP POST."""
                import http.client

                try:
                    conn = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=5)
                    conn.request("POST", "/clear")
                    conn.getresponse()
                    conn.close()
                except Exception:
                    pass

        # Return a wrapper that points to the running collector
        wrapper = ExistingCollectorWrapper(endpoint)
        yield wrapper
    else:
        manager = InMemoryTelemetryManager()
        try:
            yield manager.collector
        finally:
            manager.shutdown()


@pytest.fixture(scope="session")
def llama_stack_client(telemetry_test_collector, request):
    """Ensure telemetry collector is ready before initializing the stack client.

    This guarantees the collector is listening before any requests are made,
    preventing telemetry data from being lost.
    """
    patch_httpx_for_test_id()
    client = instantiate_llama_stack_client(request.session)
    return client


@pytest.fixture
def mock_otlp_collector(telemetry_test_collector):
    """Provides access to telemetry data and clears between tests."""
    telemetry_test_collector.clear()
    try:
        yield telemetry_test_collector
    finally:
        telemetry_test_collector.clear()
