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
            api=Api.safety,
            provider_type="inline::prompt-guard",
            pip_packages=[
                "transformers[accelerate]",
                "torch --index-url https://download.pytorch.org/whl/cpu",
            ],
            module="llama_stack.providers.inline.safety.prompt_guard",
            config_class="llama_stack.providers.inline.safety.prompt_guard.PromptGuardConfig",
            description="Prompt Guard safety provider for detecting and filtering unsafe prompts and content.",
        ),
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::llama-guard",
            pip_packages=[],
            module="llama_stack.providers.inline.safety.llama_guard",
            config_class="llama_stack.providers.inline.safety.llama_guard.LlamaGuardConfig",
            api_dependencies=[
                Api.inference,
            ],
            description="Llama Guard safety provider for content moderation and safety filtering using Meta's Llama Guard model.",
        ),
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::code-scanner",
            pip_packages=[
                "codeshield",
            ],
            module="llama_stack.providers.inline.safety.code_scanner",
            config_class="llama_stack.providers.inline.safety.code_scanner.CodeScannerConfig",
            description="Code Scanner safety provider for detecting security vulnerabilities and unsafe code patterns.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="bedrock",
            provider_type="remote::bedrock",
            pip_packages=["boto3"],
            module="llama_stack.providers.remote.safety.bedrock",
            config_class="llama_stack.providers.remote.safety.bedrock.BedrockSafetyConfig",
            description="AWS Bedrock safety provider for content moderation using AWS's safety services.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="nvidia",
            provider_type="remote::nvidia",
            pip_packages=["requests"],
            module="llama_stack.providers.remote.safety.nvidia",
            config_class="llama_stack.providers.remote.safety.nvidia.NVIDIASafetyConfig",
            description="NVIDIA's safety provider for content moderation and safety filtering.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="sambanova",
            provider_type="remote::sambanova",
            pip_packages=["litellm", "requests"],
            module="llama_stack.providers.remote.safety.sambanova",
            config_class="llama_stack.providers.remote.safety.sambanova.SambaNovaSafetyConfig",
            provider_data_validator="llama_stack.providers.remote.safety.sambanova.config.SambaNovaProviderDataValidator",
            description="SambaNova's safety provider for content moderation and safety filtering.",
        ),
    ]
