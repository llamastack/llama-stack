# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal

from pydantic import BaseModel, Field


class OTelConfig(BaseModel):
    """
    OpenTelemetry instrumentation configuration.

    Most OTel settings use environment variables (OTEL_*).
    See: https://opentelemetry.io/docs/specs/otel/configuration/sdk-configuration-variables/
    """

    service_name: str | None = Field(
        default=None,
        description="Service name (overridden by OTEL_SERVICE_NAME env var)",
    )
    span_processor: Literal["batch", "simple"] = Field(
        default="batch",
        description="Span processor type (overridden by OTEL_SPAN_PROCESSOR env var)",
    )
