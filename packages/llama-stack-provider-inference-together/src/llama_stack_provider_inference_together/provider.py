# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="together",
        provider_type="remote::together",
        pip_packages=[],
        module="llama_stack_provider_inference_together",
        config_class="llama_stack_provider_inference_together.TogetherImplConfig",
        provider_data_validator="llama_stack_provider_inference_together.TogetherProviderDataValidator",
        description="Together AI inference provider for open-source models and collaborative AI development.",
    )
