# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="fireworks",
        provider_type="remote::fireworks",
        pip_packages=[],
        module="llama_stack_provider_inference_fireworks",
        config_class="llama_stack_provider_inference_fireworks.FireworksImplConfig",
        provider_data_validator="llama_stack_provider_inference_fireworks.FireworksProviderDataValidator",
        description="Fireworks AI inference provider for Llama models and other AI models on the Fireworks platform.",
    )
