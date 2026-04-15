# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec, ProviderSpec


def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.responses,
        provider_type="inline::builtin",
        pip_packages=[],
        module="llama_stack_provider_responses_builtin",
        config_class="llama_stack_provider_responses_builtin.config.BuiltinResponsesImplConfig",
        api_dependencies=[
            Api.inference,
            Api.vector_io,
            Api.tool_runtime,
            Api.tool_groups,
            Api.conversations,
            Api.prompts,
            Api.files,
            Api.connectors,
        ],
        optional_api_dependencies=[Api.safety],
        description="Meta's reference implementation of an agent system that can use tools, access vector databases, and perform complex reasoning tasks.",
    )
