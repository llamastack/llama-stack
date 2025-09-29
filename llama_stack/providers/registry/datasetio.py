# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.datasetio,
            provider_type="inline::localfs",
            pip_packages=["pandas"],
            module="llama_stack.providers.inline.datasetio.localfs",
            config_class="llama_stack.providers.inline.datasetio.localfs.LocalFSDatasetIOConfig",
            api_dependencies=[],
            description="Local filesystem-based dataset I/O provider for reading and writing datasets to local storage.",
        ),
        RemoteProviderSpec(
            api=Api.datasetio,
            adapter_type="huggingface",
            provider_type="remote::huggingface",
            pip_packages=[
                "datasets>=4.0.0",
            ],
            module="llama_stack.providers.remote.datasetio.huggingface",
            config_class="llama_stack.providers.remote.datasetio.huggingface.HuggingfaceDatasetIOConfig",
            description="HuggingFace datasets provider for accessing and managing datasets from the HuggingFace Hub.",
        ),
        RemoteProviderSpec(
            api=Api.datasetio,
            adapter_type="nvidia",
            provider_type="remote::nvidia",
            module="llama_stack.providers.remote.datasetio.nvidia",
            config_class="llama_stack.providers.remote.datasetio.nvidia.NvidiaDatasetIOConfig",
            pip_packages=[
                "datasets>=4.0.0",
            ],
            description="NVIDIA's dataset I/O provider for accessing datasets from NVIDIA's data platform.",
        ),
    ]
