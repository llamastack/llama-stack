# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .models import MODEL_ENTRIES


class GroqInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    _config: GroqConfig

    def __init__(self, config: GroqConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            litellm_provider_name="groq",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="groq_api_key",
        )
        self.config = config

    # Delegate the client data handling get_api_key method to LiteLLMOpenAIMixin
    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        return f"{self.config.url}/openai/v1"

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()
