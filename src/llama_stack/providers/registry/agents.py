# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.core.storage.kvstore import kvstore_dependencies
from llama_stack_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.agents,
            provider_type="inline::meta-reference",
            pip_packages=[
                "matplotlib",
                "fonttools>=4.60.2",
                "pillow",
                "pandas",
                "scikit-learn",
                "mcp>=1.23.0",
            ]
            + kvstore_dependencies(),  # TODO make this dynamic based on the kvstore config
            module="llama_stack.providers.inline.agents.meta_reference",
            config_class="llama_stack.providers.inline.agents.meta_reference.MetaReferenceAgentsImplConfig",
            api_dependencies=[
                Api.inference,
                Api.vector_io,
                Api.tool_runtime,
                Api.tool_groups,
                Api.conversations,
                Api.prompts,
                Api.files,
            ],
            optional_api_dependencies=[
                Api.safety,
            ],
            description="Meta's reference implementation of an agent system that can use tools, access vector databases, and perform complex reasoning tasks.",
        ),
    ]
