# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from .config import GeminiConfig
from .models import MODEL_ENTRIES


class GeminiInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: GeminiConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            litellm_provider_name="gemini",
            api_key_from_config=config.api_key.get_secret_value() if config.api_key else None,
            provider_data_api_key_field="gemini_api_key",
        )
        self.config = config

    async def initialize(self) -> None:
        await super().initialize()

    async def shutdown(self) -> None:
        await super().shutdown()
