# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="nvidia",
        provider_type="remote::nvidia",
        pip_packages=[],
        module="llama_stack_provider_inference_nvidia",
        config_class="llama_stack_provider_inference_nvidia.NVIDIAConfig",
        provider_data_validator="llama_stack_provider_inference_nvidia.config.NVIDIAProviderDataValidator",
        description="NVIDIA inference provider for accessing NVIDIA NIM models and AI services.",
    )
