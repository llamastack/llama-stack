# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.safety,
        provider_type="remote::bedrock",
        adapter_type="bedrock",
        pip_packages=[],
        module="llama_stack_provider_safety_bedrock",
        config_class="llama_stack_provider_safety_bedrock.config.BedrockSafetyConfig",
        description="AWS Bedrock safety provider for content moderation using AWS's safety services.",
    )
