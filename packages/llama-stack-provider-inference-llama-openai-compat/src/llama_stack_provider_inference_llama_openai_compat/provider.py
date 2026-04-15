# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="llama-openai-compat",
        provider_type="remote::llama-openai-compat",
        pip_packages=[],
        module="llama_stack_provider_inference_llama_openai_compat",
        config_class="llama_stack_provider_inference_llama_openai_compat.config.LlamaCompatConfig",
        provider_data_validator="llama_stack_provider_inference_llama_openai_compat.config.LlamaProviderDataValidator",
        description="Llama OpenAI-compatible provider for using Llama models with OpenAI API format.",
    )
