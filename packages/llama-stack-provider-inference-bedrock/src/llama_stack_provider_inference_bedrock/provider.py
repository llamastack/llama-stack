# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="bedrock",
        provider_type="remote::bedrock",
        pip_packages=[],
        module="llama_stack_provider_inference_bedrock",
        config_class="llama_stack_provider_inference_bedrock.BedrockConfig",
        provider_data_validator="llama_stack_provider_inference_bedrock.config.BedrockProviderDataValidator",
        description="AWS Bedrock inference provider using OpenAI compatible endpoint.",
    )
