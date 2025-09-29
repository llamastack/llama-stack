# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import AnthropicConfig


class AnthropicInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    # source: https://docs.claude.com/en/docs/build-with-claude/embeddings
    # TODO: add support for voyageai, which is where these models are hosted
    # embedding_model_metadata = {
    #     "voyage-3-large": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-3.5": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-3.5-lite": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-code-3": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-finance-2": {"embedding_dimension": 1024, "context_length": 32000},
    #     "voyage-law-2": {"embedding_dimension": 1024, "context_length": 16000},
    #     "voyage-multimodal-3": {"embedding_dimension": 1024, "context_length": 32000},
    # }

    def __init__(self, config: AnthropicConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
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
