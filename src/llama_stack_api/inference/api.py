# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from llama_stack_api.models import Model

from .models import (
    GetChatCompletionRequest,
    ListChatCompletionsRequest,
    ListOpenAIChatCompletionResponse,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankRequest,
    RerankResponse,
)


class ModelStore(Protocol):
    async def get_model(self, identifier: str) -> Model: ...


@runtime_checkable
class InferenceProvider(Protocol):
    """
    This protocol defines the interface that should be implemented by all inference providers.
    """

    API_NAMESPACE: str = "Inference"

    model_store: ModelStore | None = None

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        """Rerank a list of documents based on their relevance to a query."""
        raise NotImplementedError("Reranking is not implemented")
        return  # this is so mypy's safe-super rule will consider the method concrete

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Generate an OpenAI-compatible completion for the given prompt using the specified model."""
        ...

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Generate an OpenAI-compatible chat completion for the given messages using the specified model."""
        ...

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Generate OpenAI-compatible embeddings for the given input using the specified model."""
        ...


class Inference(InferenceProvider):
    """Inference

    Llama Stack Inference API for generating completions, chat completions, and embeddings.

    This API provides the raw interface to the underlying models. Three kinds of models are supported:
    - LLM models: these models generate "raw" and "chat" (conversational) completions.
    - Embedding models: these models generate embeddings to be used for semantic search.
    - Rerank models: these models reorder the documents based on their relevance to a query.
    """

    async def list_chat_completions(
        self,
        request: ListChatCompletionsRequest,
    ) -> ListOpenAIChatCompletionResponse:
        """List stored chat completions."""
        raise NotImplementedError("List chat completions is not implemented")

    async def get_chat_completion(self, request: GetChatCompletionRequest) -> OpenAICompletionWithInputMessages:
        """Retrieve a stored chat completion by its ID."""
        raise NotImplementedError("Get chat completion is not implemented")
