# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.safety,
        provider_type="inline::prompt-guard",
        # Dependencies are managed by this package's pyproject.toml and
        # installed automatically via the uv workspace.
        pip_packages=[],
        module="llama_stack_provider_safety_prompt_guard",
        config_class="llama_stack_provider_safety_prompt_guard.config.PromptGuardConfig",
        description="Prompt Guard safety provider for detecting and filtering unsafe prompts and content.",
    )
