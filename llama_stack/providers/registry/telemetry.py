# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.telemetry,
            provider_type="inline::meta-reference",
            pip_packages=[
                "opentelemetry-sdk",
                "opentelemetry-exporter-otlp-proto-http",
            ],
            optional_api_dependencies=[Api.datasetio],
            module="llama_stack.providers.inline.telemetry.meta_reference",
            config_class="llama_stack.providers.inline.telemetry.meta_reference.config.TelemetryConfig",
            description="Meta's reference implementation of telemetry and observability using OpenTelemetry.",
        ),
    ]
