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
from tests.integration.telemetry.collectors import InMemoryTelemetryManager, OtlpHttpTestCollector


@pytest.fixture(scope="session")
def telemetry_test_collector():
    stack_mode = os.environ.get("LLAMA_STACK_TEST_STACK_CONFIG_TYPE", "library_client")

    if stack_mode == "server":
        # In server mode, the collector must be started and the server is already running.
        # The integration test script (scripts/integration-tests.sh) should have set
        # LLAMA_STACK_TEST_COLLECTOR_PORT and OTEL_EXPORTER_OTLP_ENDPOINT before starting the server.
        try:
            collector = OtlpHttpTestCollector()
        except RuntimeError as exc:
            pytest.skip(str(exc))

        # Verify the collector is listening on the expected endpoint
        expected_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if expected_endpoint and collector.endpoint != expected_endpoint:
            pytest.skip(
                f"Collector endpoint mismatch: expected {expected_endpoint}, got {collector.endpoint}. "
                "Server was likely started before collector."
            )

        try:
            yield collector
        finally:
            collector.shutdown()
    else:
        manager = InMemoryTelemetryManager()
        try:
            yield manager.collector
        finally:
            manager.shutdown()


@pytest.fixture(scope="session")
def llama_stack_client(telemetry_test_collector, request):
    """Ensure telemetry collector is ready before initializing the stack client."""
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
