# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Annotated, Protocol, runtime_checkable

from fastapi import Body

from llama_stack.apis.common.responses import Order
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import (
    ListOpenAIChatCompletionResponse,
    ModelStore,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankResponse,
)


@runtime_checkable
@trace_protocol
class InferenceService(Protocol):
    """
    This protocol defines the interface that should be implemented by all inference providers.

    Llama Stack Inference API for generating completions, chat completions, and embeddings.

    This API provides the raw interface to the underlying models. Three kinds of models are supported:
    - LLM models: these models generate "raw" and "chat" (conversational) completions.
    - Embedding models: these models generate embeddings to be used for semantic search.
    - Rerank models: these models reorder the documents based on their relevance to a query.
    """

    API_NAMESPACE: str = "Inference"

    model_store: ModelStore | None = None

    async def rerank(
        self,
        model: str,
        query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam],
        max_num_results: int | None = None,
    ) -> RerankResponse:
        """Rerank a list of documents based on their relevance to a query."""
        ...

    async def openai_completion(
        self,
        params: Annotated[OpenAICompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAICompletion:
        """Create completion."""
        ...

    async def openai_chat_completion(
        self,
        params: Annotated[OpenAIChatCompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Create chat completions."""
        ...

    async def openai_embeddings(
        self,
        params: Annotated[OpenAIEmbeddingsRequestWithExtraBody, Body(...)],
    ) -> OpenAIEmbeddingsResponse:
        """Create embeddings."""
        ...

    async def list_chat_completions(
        self,
        after: str | None = None,
        limit: int | None = 20,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIChatCompletionResponse:
        """List chat completions."""
        ...

    async def get_chat_completion(self, completion_id: str) -> OpenAICompletionWithInputMessages:
        """Get chat completion."""
        ...
