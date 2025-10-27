# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry test configuration supporting both library and server test modes."""

import os

import pytest

import llama_stack.core.telemetry.telemetry as telemetry_module
from llama_stack.testing.api_recorder import patch_httpx_for_test_id
from tests.integration.fixtures.common import instantiate_llama_stack_client
from tests.integration.telemetry.collectors import InMemoryTelemetryManager, OtlpHttpTestCollector


@pytest.fixture(scope="session")
def telemetry_test_collector():
    stack_mode = os.environ.get("LLAMA_STACK_TEST_STACK_CONFIG_TYPE", "library_client")

    if stack_mode == "server":
        try:
            collector = OtlpHttpTestCollector()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        env_overrides = {
            "OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint,
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_BSP_SCHEDULE_DELAY": "200",
            "OTEL_BSP_EXPORT_TIMEOUT": "2000",
        }

        previous_env = {key: os.environ.get(key) for key in env_overrides}
        previous_force_restart = os.environ.get("LLAMA_STACK_TEST_FORCE_SERVER_RESTART")

        for key, value in env_overrides.items():
            os.environ[key] = value

        os.environ["LLAMA_STACK_TEST_FORCE_SERVER_RESTART"] = "1"
        telemetry_module._TRACER_PROVIDER = None

        try:
            yield collector
        finally:
            collector.shutdown()
            for key, prior in previous_env.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior
            if previous_force_restart is None:
                os.environ.pop("LLAMA_STACK_TEST_FORCE_SERVER_RESTART", None)
            else:
                os.environ["LLAMA_STACK_TEST_FORCE_SERVER_RESTART"] = previous_force_restart
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
