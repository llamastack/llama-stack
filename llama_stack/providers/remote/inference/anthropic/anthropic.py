# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import AnthropicConfig
from .models import MODEL_ENTRIES


class AnthropicInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    def __init__(self, config: AnthropicConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            litellm_provider_name="anthropic",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="anthropic_api_key",
        )
        self.config = config

    async def initialize(self) -> None:
        await super().initialize()

    async def shutdown(self) -> None:
        await super().shutdown()

    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self):
        return "https://api.anthropic.com/v1"
