# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="watsonx",
        provider_type="remote::watsonx",
        pip_packages=[],
        module="llama_stack_provider_inference_watsonx",
        config_class="llama_stack_provider_inference_watsonx.WatsonXConfig",
        provider_data_validator="llama_stack_provider_inference_watsonx.config.WatsonXProviderDataValidator",
        description="IBM WatsonX inference provider for accessing AI models on IBM's WatsonX platform.",
    )
