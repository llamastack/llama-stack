# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

import llama_stack.providers.inline.telemetry.meta_reference.telemetry as telemetry_module
from llama_stack.log import get_logger
from llama_stack.providers.inline.telemetry.meta_reference.config import (
    TelemetryConfig,
    TelemetrySink,
)

logger = get_logger(name=__name__, category="telemetry_test_meta_reference")


def _reset_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the telemetry module re-runs initialization code that emits warnings
    monkeypatch.setattr(telemetry_module, "_TRACER_PROVIDER", None, raising=False)


def _make_config_with_sinks(*sinks: TelemetrySink) -> TelemetryConfig:
    return TelemetryConfig(sinks=list(sinks))


def _otel_logger_records(caplog: pytest.LogCaptureFixture):
    module_logger_name = "llama_stack.providers.inline.telemetry.meta_reference.telemetry"
    return [r for r in caplog.records if r.name == module_logger_name]


def test_warns_when_traces_endpoints_missing(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    _reset_provider(monkeypatch)
    # Remove both endpoints to simulate incorrect configuration
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    caplog.set_level("WARNING")

    config = _make_config_with_sinks(TelemetrySink.OTEL_TRACE)
    telemetry_module.TelemetryAdapter(config=config, deps={})

    messages = [r.getMessage() for r in _otel_logger_records(caplog)]
    assert any(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT is not set. Traces will not be exported."
        in m
        for m in messages
    )


def test_warns_when_metrics_endpoints_missing(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    _reset_provider(monkeypatch)
    # Remove both endpoints to simulate incorrect configuration
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    caplog.set_level("WARNING")

    config = _make_config_with_sinks(TelemetrySink.OTEL_METRIC)
    telemetry_module.TelemetryAdapter(config=config, deps={})

    messages = [r.getMessage() for r in _otel_logger_records(caplog)]
    assert any(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT is not set. Metrics will not be exported."
        in m
        for m in messages
    )


def test_no_warning_when_traces_endpoints_present(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    _reset_provider(monkeypatch)
    # Both must be present per current implementation to avoid warnings
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://otel.example:4318/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.example:4318")

    caplog.set_level("WARNING")

    config = _make_config_with_sinks(TelemetrySink.OTEL_TRACE)
    telemetry_module.TelemetryAdapter(config=config, deps={})

    messages = [r.getMessage() for r in _otel_logger_records(caplog)]
    assert not any("Traces will not be exported." in m for m in messages)


def test_no_warning_when_metrics_endpoints_present(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    _reset_provider(monkeypatch)
    # Both must be present per current implementation to avoid warnings
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "https://otel.example:4318/v1/metrics")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.example:4318")

    caplog.set_level("WARNING")

    config = _make_config_with_sinks(TelemetrySink.OTEL_METRIC)
    telemetry_module.TelemetryAdapter(config=config, deps={})

    messages = [r.getMessage() for r in _otel_logger_records(caplog)]
    assert not any("Metrics will not be exported." in m for m in messages)
