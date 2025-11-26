# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec, ProviderSpec, RemoteProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.prompts,
            provider_type="inline::reference",
            pip_packages=[],
            module="llama_stack.providers.inline.prompts.reference",
            config_class="llama_stack.providers.inline.prompts.reference.ReferencePromptsConfig",
            description="Reference implementation storing prompts in KVStore (SQLite, PostgreSQL, etc.)",
        ),
        RemoteProviderSpec(
            api=Api.prompts,
            adapter_type="mlflow",
            provider_type="remote::mlflow",
            pip_packages=["mlflow>=3.4.0"],
            module="llama_stack.providers.remote.prompts.mlflow",
            config_class="llama_stack.providers.remote.prompts.mlflow.MLflowPromptsConfig",
            provider_data_validator="llama_stack.providers.remote.prompts.mlflow.config.MLflowProviderDataValidator",
            description="MLflow Prompt Registry provider for centralized prompt management and versioning",
        ),
    ]
