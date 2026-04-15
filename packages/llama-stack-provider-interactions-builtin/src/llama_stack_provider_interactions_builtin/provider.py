# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec, ProviderSpec


def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.interactions,
        provider_type="inline::builtin",
        pip_packages=[],
        module="llama_stack_provider_interactions_builtin",
        config_class="llama_stack_provider_interactions_builtin.config.InteractionsConfig",
        api_dependencies=[Api.inference],
        description="Serves the Google Interactions API so that Google GenAI SDK and ADK clients can call Llama Stack without code changes.",
    )
