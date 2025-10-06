# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.apis.inference.inference import OpenAICompletion, OpenAIEmbeddingsResponse
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

    def get_api_key(self) -> str:
        return self.config.api_key or ""

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The Llama API base URL
        """
        return self.config.openai_compat_api_base

    async def openai_completion(
        self,
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        suffix: str | None = None,
    ) -> OpenAICompletion:
        raise NotImplementedError()

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
