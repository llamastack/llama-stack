# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Inference API.

This module defines the FastAPI router for the Inference API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

import inspect
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_stack_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1

from .api import Inference
from .models import (
    GetChatCompletionRequest,
    ListChatCompletionsRequest,
    ListOpenAIChatCompletionResponse,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankRequest,
    RerankResponse,
)

# Automatically generate dependency functions from Pydantic models
# This ensures the models are the single source of truth for descriptions
get_list_chat_completions_request = create_query_dependency(ListChatCompletionsRequest)
get_chat_completion_request = create_path_dependency(GetChatCompletionRequest)


def _create_sse_event(data: Any) -> str:
    """Create a Server-Sent Event formatted string from data."""
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)
    return f"data: {data}\n\n"


async def _sse_generator(event_gen_or_coroutine: Any) -> AsyncIterator[str]:
    """Convert an async generator (or coroutine returning one) to SSE format.

    This handles both direct async generators and coroutines that return async generators.
    """
    # If it's a coroutine, await it to get the async generator
    if inspect.iscoroutine(event_gen_or_coroutine):
        event_gen = await event_gen_or_coroutine
    else:
        event_gen = event_gen_or_coroutine

    # Now iterate the async generator and yield SSE events
    async for item in event_gen:
        yield _create_sse_event(item)


def create_router(impl: Inference) -> APIRouter:
    """Create a FastAPI router for the Inference API.

    Args:
        impl: The Inference implementation instance

    Returns:
        APIRouter configured for the Inference API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Inference"],
        responses=standard_responses,
    )

    @router.post(
        "/chat/completions",
        response_model=None,  # Dynamic response: non-streaming (JSON) or streaming (SSE)
        summary="Create chat completions.",
        description="Generate an OpenAI-compatible chat completion for the given messages using the specified model.",
        responses={
            200: {
                "description": "An OpenAIChatCompletion. When streaming, returns Server-Sent Events (SSE) with OpenAIChatCompletionChunk objects.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/OpenAIChatCompletion"}},
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/OpenAIChatCompletionChunk"}},
                },
            },
        },
    )
    async def openai_chat_completion(
        params: Annotated[OpenAIChatCompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAIChatCompletion | StreamingResponse:
        result = impl.openai_chat_completion(params)
        if params.stream:
            return StreamingResponse(_sse_generator(result), media_type="text/event-stream")
        return await result

    @router.get(
        "/chat/completions",
        response_model=ListOpenAIChatCompletionResponse,
        summary="List chat completions.",
        description="List chat completions.",
        responses={
            200: {"description": "A ListOpenAIChatCompletionResponse."},
        },
    )
    async def list_chat_completions(
        request: Annotated[ListChatCompletionsRequest, Depends(get_list_chat_completions_request)],
    ) -> ListOpenAIChatCompletionResponse:
        return await impl.list_chat_completions(request)

    @router.get(
        "/chat/completions/{completion_id}",
        response_model=OpenAICompletionWithInputMessages,
        summary="Get chat completion.",
        description="Describe a chat completion by its ID.",
        responses={
            200: {"description": "A OpenAICompletionWithInputMessages."},
        },
    )
    async def get_chat_completion(
        request: Annotated[GetChatCompletionRequest, Depends(get_chat_completion_request)],
    ) -> OpenAICompletionWithInputMessages:
        return await impl.get_chat_completion(request)

    @router.post(
        "/completions",
        response_model=None,  # Dynamic response: non-streaming (JSON) or streaming (SSE)
        summary="Create completion.",
        description="Generate an OpenAI-compatible completion for the given prompt using the specified model.",
        responses={
            200: {
                "description": "An OpenAICompletion. When streaming, returns Server-Sent Events (SSE) with OpenAICompletion chunks.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/OpenAICompletion"}},
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/OpenAICompletion"}},
                },
            },
        },
    )
    async def openai_completion(
        params: Annotated[OpenAICompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAICompletion | StreamingResponse:
        result = impl.openai_completion(params)
        if params.stream:
            return StreamingResponse(_sse_generator(result), media_type="text/event-stream")
        return await result

    @router.post(
        "/embeddings",
        response_model=OpenAIEmbeddingsResponse,
        summary="Create embeddings.",
        description="Generate OpenAI-compatible embeddings for the given input using the specified model.",
        responses={
            200: {"description": "An OpenAIEmbeddingsResponse containing the embeddings."},
        },
    )
    async def openai_embeddings(
        params: Annotated[OpenAIEmbeddingsRequestWithExtraBody, Body(...)],
    ) -> OpenAIEmbeddingsResponse:
        return await impl.openai_embeddings(params)

    @router.post(
        "/inference/rerank",
        response_model=RerankResponse,
        summary="Rerank documents based on relevance to a query.",
        description="Rerank a list of documents based on their relevance to a query.",
        responses={
            200: {"description": "RerankResponse with indices sorted by relevance score (descending)."},
        },
    )
    async def rerank(
        request: Annotated[RerankRequest, Body(...)],
    ) -> RerankResponse:
        return await impl.rerank(request)

    return router
