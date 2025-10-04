# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

type BatchSpanProcessor = Literal["batch"]
type SimpleSpanProcessor = Literal["simple"]


class OTelTelemetryConfig(BaseModel):
    """
    The configuration for the OpenTelemetry telemetry provider.
    Most configuration is set using environment variables.
    See https://opentelemetry.io/docs/specs/otel/configuration/sdk-configuration-variables/ for more information.
    """

    service_name: str = Field(
        description="""The name of the service to be monitored.
        Is overridden by the OTEL_SERVICE_NAME or OTEL_RESOURCE_ATTRIBUTES environment variables.""",
    )
    service_version: str | None = Field(
        default=None,
        description="""The version of the service to be monitored.
        Is overriden by the OTEL_RESOURCE_ATTRIBUTES environment variable.""",
    )
    deployment_environment: str | None = Field(
        default=None,
        description="""The name of the environment of the service to be monitored.
        Is overriden by the OTEL_RESOURCE_ATTRIBUTES environment variable.""",
    )
    span_processor: BatchSpanProcessor | SimpleSpanProcessor | None = Field(
        description="""The span processor to use.
        Is overriden by the OTEL_SPAN_PROCESSOR environment variable.""",
        default="batch",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "") -> dict[str, Any]:
        """Sample configuration for use in distributions."""
        return {
            "service_name": "${env.OTEL_SERVICE_NAME:=llama-stack}",
            "service_version": "${env.OTEL_SERVICE_VERSION:=}",
            "deployment_environment": "${env.OTEL_DEPLOYMENT_ENVIRONMENT:=}",
            "span_processor": "${env.OTEL_SPAN_PROCESSOR:=batch}",
        }
