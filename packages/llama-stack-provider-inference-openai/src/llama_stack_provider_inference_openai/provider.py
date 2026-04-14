# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="openai",
        provider_type="remote::openai",
        pip_packages=[],
        module="llama_stack_provider_inference_openai",
        config_class="llama_stack_provider_inference_openai.OpenAIConfig",
        provider_data_validator="llama_stack_provider_inference_openai.config.OpenAIProviderDataValidator",
        description="OpenAI inference provider for accessing GPT models and other OpenAI services.",
    )
