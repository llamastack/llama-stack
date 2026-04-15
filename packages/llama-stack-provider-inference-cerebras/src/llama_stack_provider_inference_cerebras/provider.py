# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="cerebras",
        provider_type="remote::cerebras",
        pip_packages=[],
        module="llama_stack_provider_inference_cerebras",
        config_class="llama_stack_provider_inference_cerebras.CerebrasImplConfig",
        provider_data_validator="llama_stack_provider_inference_cerebras.config.CerebrasProviderDataValidator",
        description="Cerebras inference provider for running models on Cerebras Cloud platform.",
    )
