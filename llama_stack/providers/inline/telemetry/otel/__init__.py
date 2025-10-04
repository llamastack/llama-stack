# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import OTelTelemetryConfig

__all__ = ["OTelTelemetryConfig"]


async def get_provider_impl(config: OTelTelemetryConfig, deps):
    """
    Get the OTel telemetry provider implementation.

    This function is called by the Llama Stack registry to instantiate
    the provider.
    """
    from .otel import OTelTelemetryProvider

    # The provider is synchronously initialized via Pydantic model_post_init
    # No async initialization needed
    return OTelTelemetryProvider(config=config)
