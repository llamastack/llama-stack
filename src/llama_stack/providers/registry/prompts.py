# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, ProviderSpec, RemoteProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        RemoteProviderSpec(
            api=Api.prompts,
            adapter_type="mlflow",
            provider_type="remote::mlflow",
            pip_packages=["mlflow>=3.4.0"],
            module="llama_stack.providers.remote.prompts.mlflow",
            config_class="llama_stack.providers.remote.prompts.mlflow.MLflowPromptsConfig",
            description="MLflow Prompt Registry provider for centralized prompt management and versioning",
        ),
    ]
