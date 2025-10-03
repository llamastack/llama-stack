# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from urllib.parse import urljoin

from cerebras.cloud.sdk import AsyncCerebras

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionRequest,
    Inference,
    OpenAIEmbeddingsResponse,
    TopKSamplingStrategy,
)
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
)

from .config import CerebrasImplConfig


class CerebrasInferenceAdapter(
    OpenAIMixin,
    Inference,
):
    def __init__(self, config: CerebrasImplConfig) -> None:
        self.config = config

        # TODO: make this use provider data, etc. like other providers
        self._cerebras_client = AsyncCerebras(
            base_url=self.config.base_url,
            api_key=self.config.api_key.get_secret_value(),
        )

    def get_api_key(self) -> str:
        return self.config.api_key.get_secret_value()

    def get_base_url(self) -> str:
        return urljoin(self.config.base_url, "v1")

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def _get_params(self, request: ChatCompletionRequest | CompletionRequest) -> dict:
        if request.sampling_params and isinstance(request.sampling_params.strategy, TopKSamplingStrategy):
            raise ValueError("`top_k` not supported by Cerebras")

        prompt = ""
        if isinstance(request, ChatCompletionRequest):
            prompt = await chat_completion_request_to_prompt(request, self.get_llama_model(request.model))
        elif isinstance(request, CompletionRequest):
            prompt = await completion_request_to_prompt(request)
        else:
            raise ValueError(f"Unknown request type {type(request)}")

        return {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
