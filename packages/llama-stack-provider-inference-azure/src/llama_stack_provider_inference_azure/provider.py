# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="azure",
        provider_type="remote::azure",
        pip_packages=[],
        module="llama_stack_provider_inference_azure",
        config_class="llama_stack_provider_inference_azure.AzureConfig",
        provider_data_validator="llama_stack_provider_inference_azure.config.AzureProviderDataValidator",
        description="Azure OpenAI inference provider for accessing GPT models and other Azure services.",
    )
