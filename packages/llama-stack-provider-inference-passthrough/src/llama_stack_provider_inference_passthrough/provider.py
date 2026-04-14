# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="passthrough",
        provider_type="remote::passthrough",
        pip_packages=[],
        module="llama_stack_provider_inference_passthrough",
        config_class="llama_stack_provider_inference_passthrough.PassthroughImplConfig",
        provider_data_validator="llama_stack_provider_inference_passthrough.PassthroughProviderDataValidator",
        description="Passthrough inference provider for connecting to any external inference service not directly supported.",
    )
