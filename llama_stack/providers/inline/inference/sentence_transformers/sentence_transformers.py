# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

from llama_stack.apis.inference import (
    InferenceProvider,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionToLlamaStackMixin,
)

from .config import SentenceTransformersInferenceConfig

log = get_logger(name=__name__, category="inference")


class SentenceTransformersInferenceImpl(
    OpenAIChatCompletionToLlamaStackMixin,
    SentenceTransformerEmbeddingMixin,
    InferenceProvider,
    ModelsProtocolPrivate,
):
    __provider_id__: str

    def __init__(self, config: SentenceTransformersInferenceConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def should_refresh_models(self) -> bool:
        return False

    async def list_models(self) -> list[Model] | None:
        return [
            Model(
                identifier="all-MiniLM-L6-v2",
                provider_resource_id="all-MiniLM-L6-v2",
                provider_id=self.__provider_id__,
                metadata={
                    "embedding_dimension": 384,
                },
                model_type=ModelType.embedding,
            ),
        ]

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def openai_completion(
        self,
        # Standard OpenAI completion parameters
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
        # vLLM-specific parameters
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        # for fill-in-the-middle type completion
        suffix: str | None = None,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion not supported by sentence transformers provider")

    async def openai_chat_completion(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        frequency_penalty: float | None = None,
        function_call: str | dict[str, Any] | None = None,
        functions: list[dict[str, Any]] | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: OpenAIResponseFormatParam | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError("OpenAI chat completion not supported by sentence transformers provider")
