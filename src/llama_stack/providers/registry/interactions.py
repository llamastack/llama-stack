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
    """Return the list of available interactions provider specifications."""
    return [
        InlineProviderSpec(
            api=Api.interactions,
            provider_type="inline::builtin",
            pip_packages=[],
            module="llama_stack.providers.inline.interactions",
            config_class="llama_stack.providers.inline.interactions.config.InteractionsConfig",
            api_dependencies=[
                Api.inference,
            ],
            description=(
                "Serves the Google Interactions API (POST /v1alpha/interactions) so that "
                "Google GenAI SDK and ADK clients can call Llama Stack without code changes. "
                "Requests are translated to OpenAI Chat Completions and routed to whichever "
                "inference provider is configured (vLLM, Ollama, OpenAI, Bedrock, etc.). "
                "When the provider is Gemini, non-streaming requests are forwarded directly "
                "to the native /v1beta/interactions endpoint, avoiding double translation."
            ),
        ),
    ]
