# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="runpod",
        provider_type="remote::runpod",
        pip_packages=[],
        module="llama_stack_provider_inference_runpod",
        config_class="llama_stack_provider_inference_runpod.RunpodImplConfig",
        provider_data_validator="llama_stack_provider_inference_runpod.config.RunpodProviderDataValidator",
        description="RunPod inference provider for running models on RunPod's cloud GPU platform.",
    )
