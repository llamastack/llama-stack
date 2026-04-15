# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.safety,
        provider_type="inline::llama-guard",
        pip_packages=[],
        module="llama_stack_provider_safety_llama_guard",
        config_class="llama_stack_provider_safety_llama_guard.config.LlamaGuardConfig",
        api_dependencies=[Api.inference],
        description="Llama Guard safety provider for content moderation using Meta's Llama Guard model.",
    )
