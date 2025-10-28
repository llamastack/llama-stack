# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference.inference import (
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

logger = get_logger(name=__name__, category="inference::llama_openai_compat")


class LlamaCompatInferenceAdapter(OpenAIMixin):
    config: LlamaCompatConfig

    provider_data_api_key_field: str = "llama_api_key"
    """
    Llama API Inference Adapter for Llama Stack.
    """

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The Llama API base URL
        """
        return self.config.openai_compat_api_base

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError()

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
