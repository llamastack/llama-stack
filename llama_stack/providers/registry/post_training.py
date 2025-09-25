# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import cast

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec, RemoteProviderSpec

# We provide two versions of these providers so that distributions can package the appropriate version of torch.
# The CPU version is used for distributions that don't have GPU support -- they result in smaller container images.
torchtune_def = dict(
    api=Api.post_training,
    pip_packages=["numpy"],
    module="llama_stack.providers.inline.post_training.torchtune",
    config_class="llama_stack.providers.inline.post_training.torchtune.TorchtunePostTrainingConfig",
    api_dependencies=[
        Api.datasetio,
        Api.datasets,
    ],
    description="TorchTune-based post-training provider for fine-tuning and optimizing models using Meta's TorchTune framework.",
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            **{  # type: ignore
                **torchtune_def,
                "provider_type": "inline::torchtune-cpu",
                "pip_packages": (
                    cast(list[str], torchtune_def["pip_packages"])
                    + ["torch torchtune>=0.5.0 torchao>=0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu"]
                ),
            },
        ),
        InlineProviderSpec(
            **{  # type: ignore
                **torchtune_def,
                "provider_type": "inline::torchtune-gpu",
                "pip_packages": (
                    cast(list[str], torchtune_def["pip_packages"]) + ["torch torchtune>=0.5.0 torchao>=0.12.0"]
                ),
            },
        ),
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::huggingface-gpu",
            pip_packages=["trl", "transformers", "peft", "datasets>=4.0.0", "torch"],
            module="llama_stack.providers.inline.post_training.huggingface",
            config_class="llama_stack.providers.inline.post_training.huggingface.HuggingFacePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
            description="HuggingFace-based post-training provider for fine-tuning models using the HuggingFace ecosystem.",
        ),
        RemoteProviderSpec(
            api=Api.post_training,
            adapter_type="nvidia",
            provider_type="remote::nvidia",
            pip_packages=["requests", "aiohttp"],
            module="llama_stack.providers.remote.post_training.nvidia",
            config_class="llama_stack.providers.remote.post_training.nvidia.NvidiaPostTrainingConfig",
            description="NVIDIA's post-training provider for fine-tuning models on NVIDIA's platform.",
        ),
    ]
