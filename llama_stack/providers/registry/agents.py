# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.agents,
            provider_type="inline::meta-reference",
            module="llama_stack.providers.inline.agents.meta_reference",
            config_class="llama_stack.providers.inline.agents.meta_reference.MetaReferenceAgentsImplConfig",
            api_dependencies=[
                Api.inference,
                Api.safety,
                Api.vector_io,
                Api.vector_dbs,
                Api.tool_runtime,
                Api.tool_groups,
            ],
            description="Meta's reference implementation of an agent system that can use tools, access vector databases, and perform complex reasoning tasks.",
        ),
    ]
