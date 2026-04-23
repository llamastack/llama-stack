# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    """Return the list of available agents provider specifications."""
    return [
        InlineProviderSpec(
            api=Api.agents,
            provider_type="inline::builtin",
            pip_packages=[],
            module="llama_stack.providers.inline.agents",
            config_class="llama_stack.providers.inline.agents.config.AgentsConfig",
            api_dependencies=[],
            description=(
                "Implements the Anthropic Managed Agents API (v1alpha) for creating and managing agent "
                "configurations with model, tools, and system prompt specifications. This provider stores "
                "agent configurations in-memory and provides CRUD operations compatible with the "
                "Claude Agent SDK."
            ),
        ),
    ]
