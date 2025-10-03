# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
End-to-end integration tests for OpenTelemetry with automatic instrumentation.

HOW THIS WORKS:
1. Starts a mock OTLP collector (HTTP server) to receive telemetry
2. Starts a mock vLLM server to handle inference requests
3. Starts REAL Llama Stack with: opentelemetry-instrument llama stack run
4. Makes REAL API calls to the stack
5. Verifies telemetry was exported to the mock collector

WHERE TO MAKE CHANGES:
- Add test cases → See TEST_CASES list below (line ~70)
- Add mock servers → See MOCK_SERVERS list in mock_servers fixture (line ~200)
- Modify mock behavior → See mocking/servers.py
- Change stack config → See llama_stack_server fixture (line ~250)
- Add assertions → See TestOTelE2EWithRealServer class (line ~370)

RUNNING THE TESTS:
- Quick (mock servers only): pytest test_otel_e2e.py::TestMockServers -v
- Full E2E (slow): pytest test_otel_e2e.py::TestOTelE2EWithRealServer -v -m slow
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import socket
import subprocess
import time
from typing import Any

import pytest
import requests
import yaml
from pydantic import BaseModel, Field

# Mock servers are in the mocking/ subdirectory
from .mocking import (
    MockOTLPCollector,
    MockServerConfig,
    MockVLLMServer,
    start_mock_servers_async,
    stop_mock_servers,
)

# ============================================================================
# DATA MODELS
# ============================================================================


class TelemetryTestCase(BaseModel):
    """
    Pydantic model defining expected telemetry for an API call.

    **TO ADD A NEW TEST CASE:** Add to TEST_CASES list below.
    """

    name: str = Field(description="Unique test case identifier")
    http_method: str = Field(description="HTTP method (GET, POST, etc.)")
    api_path: str = Field(description="API path (e.g., '/v1/models')")
    request_body: dict[str, Any] | None = Field(default=None)
    expected_http_status: int = Field(default=200)
    expected_trace_exports: int = Field(default=1, description="Minimum number of trace exports expected")
    expected_metric_exports: int = Field(default=0, description="Minimum number of metric exports expected")
    should_have_error_span: bool = Field(default=False)


# ============================================================================
# TEST CONFIGURATION
# **TO ADD NEW TESTS:** Add TelemetryTestCase instances here
# ============================================================================

TEST_CASES = [
    TelemetryTestCase(
        name="models_list",
        http_method="GET",
        api_path="/v1/models",
        expected_trace_exports=1,
        expected_metric_exports=1,  # HTTP metrics from OTel provider middleware
    ),
    TelemetryTestCase(
        name="chat_completion",
        http_method="POST",
        api_path="/v1/inference/chat_completion",
        request_body={
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        expected_trace_exports=2,  # Stack request + vLLM backend call
        expected_metric_exports=1,  # HTTP metrics (duration, count, active_requests)
    ),
]


# ============================================================================
# TEST INFRASTRUCTURE
# ============================================================================


class TelemetryTestRunner:
    """
    Executes TelemetryTestCase instances against real Llama Stack.

    **HOW IT WORKS:**
    1. Makes real HTTP request to the stack
    2. Waits for telemetry export
    3. Verifies exports were sent to mock collector
    """

    def __init__(self, base_url: str, collector: MockOTLPCollector):
        self.base_url = base_url
        self.collector = collector

    def run_test_case(self, test_case: TelemetryTestCase, verbose: bool = False) -> bool:
        """Execute a single test case and verify telemetry."""
        initial_traces = self.collector.get_trace_count()
        initial_metrics = self.collector.get_metric_count()

        if verbose:
            print(f"\n--- {test_case.name} ---")
            print(f"  {test_case.http_method} {test_case.api_path}")

        # Make real HTTP request to Llama Stack
        try:
            url = f"{self.base_url}{test_case.api_path}"

            if test_case.http_method == "GET":
                response = requests.get(url, timeout=5)
            elif test_case.http_method == "POST":
                response = requests.post(url, json=test_case.request_body or {}, timeout=5)
            else:
                response = requests.request(test_case.http_method, url, timeout=5)

            if verbose:
                print(f"  HTTP Response: {response.status_code}")

            status_match = response.status_code == test_case.expected_http_status

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"  Request failed: {e}")
            status_match = False

        # Wait for automatic instrumentation to export telemetry
        # Traces export immediately, metrics export every 1 second (configured via env var)
        time.sleep(2.0)  # Wait for both traces and metrics to export

        # Verify traces were exported to mock collector
        new_traces = self.collector.get_trace_count() - initial_traces
        traces_exported = new_traces >= test_case.expected_trace_exports

        # Verify metrics were exported (if expected)
        new_metrics = self.collector.get_metric_count() - initial_metrics
        metrics_exported = new_metrics >= test_case.expected_metric_exports

        if verbose:
            print(
                f"  Expected: >={test_case.expected_trace_exports} trace exports, >={test_case.expected_metric_exports} metric exports"
            )
            print(f"  Actual: {new_traces} trace exports, {new_metrics} metric exports")
            result = status_match and traces_exported and metrics_exported
            print(f"  Result: {'PASS' if result else 'FAIL'}")

        return status_match and traces_exported and metrics_exported

    def run_all_test_cases(self, test_cases: list[TelemetryTestCase], verbose: bool = True) -> dict[str, bool]:
        """Run all test cases and return results."""
        results = {}
        for test_case in test_cases:
            results[test_case.name] = self.run_test_case(test_case, verbose=verbose)
        return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_port_available(port: int) -> bool:
    """Check if a TCP port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("localhost", port))
            return True
    except OSError:
        return False


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def mock_servers():
    """
    Fixture: Start all mock servers in parallel using async harness.

    **TO ADD A NEW MOCK SERVER:**
    Just add a MockServerConfig to the MOCK_SERVERS list below.
    """
    import asyncio

    # ========================================================================
    # MOCK SERVER CONFIGURATION
    # **TO ADD A NEW MOCK:** Just add a MockServerConfig instance below
    #
    # Example:
    #   MockServerConfig(
    #       name="Mock MyService",
    #       server_class=MockMyService,  # Must inherit from MockServerBase
    #       init_kwargs={"port": 9000, "param": "value"},
    #   ),
    # ========================================================================
    mock_servers_config = [
        MockServerConfig(
            name="Mock OTLP Collector",
            server_class=MockOTLPCollector,
            init_kwargs={"port": 4318},
        ),
        MockServerConfig(
            name="Mock vLLM Server",
            server_class=MockVLLMServer,
            init_kwargs={
                "port": 8000,
                "models": ["meta-llama/Llama-3.2-1B-Instruct"],
            },
        ),
        # Add more mock servers here - they will start in parallel automatically!
    ]

    # Start all servers in parallel
    servers = asyncio.run(start_mock_servers_async(mock_servers_config))

    # Verify vLLM models
    models_response = requests.get("http://localhost:8000/v1/models", timeout=1)
    models_data = models_response.json()
    print(f"[INFO] Mock vLLM serving {len(models_data['data'])} models: {[m['id'] for m in models_data['data']]}")

    yield servers

    # Stop all servers
    stop_mock_servers(servers)


@pytest.fixture(scope="module")
def mock_otlp_collector(mock_servers):
    """Convenience fixture to get OTLP collector from mock_servers."""
    return mock_servers["Mock OTLP Collector"]


@pytest.fixture(scope="module")
def mock_vllm_server(mock_servers):
    """Convenience fixture to get vLLM server from mock_servers."""
    return mock_servers["Mock vLLM Server"]


@pytest.fixture(scope="module")
def llama_stack_server(tmp_path_factory, mock_otlp_collector, mock_vllm_server):
    """
    Fixture: Start real Llama Stack server with automatic OTel instrumentation.

    **THIS IS THE MAIN FIXTURE** - it runs:
        opentelemetry-instrument llama stack run --config run.yaml

    **TO MODIFY STACK CONFIG:** Edit run_config dict below
    """
    config_dir = tmp_path_factory.mktemp("otel-stack-config")

    # Ensure mock vLLM is ready and accessible before starting Llama Stack
    print("\n[INFO] Verifying mock vLLM is accessible at http://localhost:8000...")
    try:
        vllm_models = requests.get("http://localhost:8000/v1/models", timeout=2)
        print(f"[INFO] Mock vLLM models endpoint response: {vllm_models.status_code}")
    except Exception as e:
        pytest.fail(f"Mock vLLM not accessible before starting Llama Stack: {e}")

    # Create run.yaml with inference provider
    # **TO ADD MORE PROVIDERS:** Add to providers dict
    run_config = {
        "image_name": "test-otel-e2e",
        "apis": ["inference"],
        "providers": {
            "inference": [
                {
                    "provider_id": "vllm",
                    "provider_type": "remote::vllm",
                    "config": {
                        "url": "http://localhost:8000/v1",
                    },
                },
            ],
        },
        "models": [
            {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "provider_id": "vllm",
            }
        ],
    }

    config_file = config_dir / "run.yaml"
    with open(config_file, "w") as f:
        yaml.dump(run_config, f)

    # Find available port for Llama Stack
    port = 5555
    while not is_port_available(port) and port < 5600:
        port += 1

    if port >= 5600:
        pytest.skip("No available ports for test server")

    # Set environment variables for OTel instrumentation
    # NOTE: These only affect the subprocess, not other tests
    env = os.environ.copy()
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"  # Ensure correct protocol
    env["OTEL_SERVICE_NAME"] = "llama-stack-e2e-test"
    env["LLAMA_STACK_PORT"] = str(port)
    env["OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED"] = "true"

    # Configure fast metric export for testing (default is 60 seconds)
    # This makes metrics export every 500ms instead of every 60 seconds
    env["OTEL_METRIC_EXPORT_INTERVAL"] = "500"  # milliseconds
    env["OTEL_METRIC_EXPORT_TIMEOUT"] = "1000"  # milliseconds

    # Disable inference recording to ensure real requests to our mock vLLM
    # This is critical - without this, Llama Stack replays cached responses
    # Safe to remove here as it only affects the subprocess environment
    if "LLAMA_STACK_TEST_INFERENCE_MODE" in env:
        del env["LLAMA_STACK_TEST_INFERENCE_MODE"]

    # Start server with automatic instrumentation
    cmd = [
        "opentelemetry-instrument",  # ← Automatic instrumentation wrapper
        "llama",
        "stack",
        "run",
        str(config_file),
        "--port",
        str(port),
    ]

    print(f"\n[INFO] Starting Llama Stack with OTel instrumentation on port {port}")
    print(f"[INFO] Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to start
    max_wait = 30
    base_url = f"http://localhost:{port}"

    for i in range(max_wait):
        try:
            response = requests.get(f"{base_url}/v1/health", timeout=1)
            if response.status_code == 200:
                print(f"[INFO] Server ready at {base_url}")
                break
        except requests.exceptions.RequestException:
            if i == max_wait - 1:
                process.terminate()
                stdout, stderr = process.communicate(timeout=5)
                pytest.fail(f"Server failed to start.\nStdout: {stdout}\nStderr: {stderr}")
            time.sleep(1)

    yield {
        "base_url": base_url,
        "port": port,
        "collector": mock_otlp_collector,
        "vllm_server": mock_vllm_server,
    }

    # Cleanup
    print("\n[INFO] Stopping Llama Stack server")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


# ============================================================================
# TESTS: End-to-End with Real Stack
# **THESE RUN SLOW** - marked with @pytest.mark.slow
# **TO ADD NEW E2E TESTS:** Add methods to this class
# ============================================================================


@pytest.mark.slow
class TestOTelE2E:
    """
    End-to-end tests with real Llama Stack server.

    These tests verify the complete flow:
    - Real Llama Stack with opentelemetry-instrument
    - Real API calls
    - Real automatic instrumentation
    - Mock OTLP collector captures exports
    """

    def test_server_starts_with_auto_instrumentation(self, llama_stack_server):
        """Verify server starts successfully with opentelemetry-instrument."""
        base_url = llama_stack_server["base_url"]

        # Try different health check endpoints
        health_endpoints = ["/health", "/v1/health", "/"]
        server_responding = False

        for endpoint in health_endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                print(f"\n[DEBUG] {endpoint} -> {response.status_code}")
                if response.status_code == 200:
                    server_responding = True
                    break
            except Exception as e:
                print(f"[DEBUG] {endpoint} failed: {e}")

        assert server_responding, f"Server not responding on any endpoint at {base_url}"

        print(f"\n[PASS] Llama Stack running with OTel at {base_url}")

    def test_all_test_cases_via_runner(self, llama_stack_server):
        """
        **MAIN TEST:** Run all TelemetryTestCase instances.

        This executes all test cases defined in TEST_CASES list.
        **TO ADD MORE TESTS:** Add to TEST_CASES at top of file
        """
        base_url = llama_stack_server["base_url"]
        collector = llama_stack_server["collector"]

        # Create test runner
        runner = TelemetryTestRunner(base_url, collector)

        # Execute all test cases
        results = runner.run_all_test_cases(TEST_CASES, verbose=True)

        # Print summary
        print(f"\n{'=' * 50}")
        print("TEST CASE SUMMARY")
        print(f"{'=' * 50}")
        passed = sum(1 for p in results.values() if p)
        total = len(results)
        print(f"Passed: {passed}/{total}\n")

        for name, result in results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name}")
        print(f"{'=' * 50}\n")
