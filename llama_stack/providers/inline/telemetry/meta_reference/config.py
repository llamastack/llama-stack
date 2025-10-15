# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TelemetrySink(StrEnum):
    OTEL_TRACE = "otel_trace"
    OTEL_METRIC = "otel_metric"
    CONSOLE = "console"


class TelemetryConfig(BaseModel):
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="The OpenTelemetry collector endpoint URL (base URL for traces, metrics, and logs). If not set, the SDK will use OTEL_EXPORTER_OTLP_ENDPOINT environment variable.",
    )
    service_name: str = Field(
        # service name is always the same, use zero-width space to avoid clutter
        default="\u200b",
        description="The service name to use for telemetry",
    )
    sinks: list[TelemetrySink] = Field(
        default_factory=list,
        description="List of telemetry sinks to enable (possible values: otel_trace, otel_metric, console)",
    )

    @field_validator("sinks", mode="before")
    @classmethod
    def validate_sinks(cls, v):
        if isinstance(v, str):
            return [TelemetrySink(sink.strip()) for sink in v.split(",")]
        return v or []

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "service_name": "${env.OTEL_SERVICE_NAME:=\u200b}",
            "sinks": "${env.TELEMETRY_SINKS:=}",
            "otel_exporter_otlp_endpoint": "${env.OTEL_EXPORTER_OTLP_ENDPOINT:=}",
        }
