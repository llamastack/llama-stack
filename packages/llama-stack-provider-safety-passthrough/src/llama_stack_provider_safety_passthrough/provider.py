# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.safety,
        provider_type="remote::passthrough",
        adapter_type="passthrough",
        pip_packages=[],
        module="llama_stack_provider_safety_passthrough",
        config_class="llama_stack_provider_safety_passthrough.config.PassthroughSafetyConfig",
        provider_data_validator="llama_stack_provider_safety_passthrough.config.PassthroughProviderDataValidator",
        description="Passthrough safety provider that forwards moderation calls to a downstream HTTP service.",
    )
