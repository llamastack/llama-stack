# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="vllm",
        provider_type="remote::vllm",
        pip_packages=[],
        module="llama_stack_provider_inference_vllm",
        config_class="llama_stack_provider_inference_vllm.VLLMInferenceAdapterConfig",
        provider_data_validator="llama_stack_provider_inference_vllm.VLLMProviderDataValidator",
        description="Remote vLLM inference provider for connecting to vLLM servers.",
    )
