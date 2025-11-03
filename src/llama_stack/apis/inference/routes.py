# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Annotated

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_stack.apis.common.responses import Order
from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .inference_service import InferenceService
from .models import (
    ListOpenAIChatCompletionResponse,
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


def get_inference_service(request: Request) -> InferenceService:
    """Dependency to get the inference service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.inference not in impls:
        raise ValueError("Inference API implementation not found")
    return impls[Api.inference]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Inference"],
    responses=standard_responses,
)

router_v1alpha = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
    tags=["Inference"],
    responses=standard_responses,
)


@router_v1alpha.post(
    "/inference/rerank",
    response_model=RerankResponse,
    summary="Rerank a list of documents.",
    description="Rerank a list of documents based on their relevance to a query.",
)
async def rerank(
    model: str = Body(..., description="The identifier of the reranking model to use."),
    query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam = Body(
        ..., description="The search query to rank items against."
    ),
    items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam] = Body(
        ..., description="List of items to rerank."
    ),
    max_num_results: int | None = Body(None, description="Maximum number of results to return. Default: returns all."),
    svc: InferenceService = Depends(get_inference_service),
) -> RerankResponse:
    """Rerank a list of documents based on their relevance to a query."""
    return await svc.rerank(model=model, query=query, items=items, max_num_results=max_num_results)


@router.post(
    "/completions",
    response_model=OpenAICompletion,
    summary="Create completion.",
    description="Create completion.",
)
async def openai_completion(
    params: OpenAICompletionRequestWithExtraBody = Body(...),
    svc: InferenceService = Depends(get_inference_service),
) -> OpenAICompletion:
    """Create completion."""
    return await svc.openai_completion(params=params)


@router.post(
    "/chat/completions",
    summary="Create chat completions.",
    description="Create chat completions.",
)
async def openai_chat_completion(
    params: OpenAIChatCompletionRequestWithExtraBody = Body(...),
    svc: InferenceService = Depends(get_inference_service),
):
    """Create chat completions."""
    response = await svc.openai_chat_completion(params=params)

    # Check if response is an async generator/iterator (streaming response)
    # Check for __aiter__ method which all async iterators have
    if hasattr(response, "__aiter__"):
        # Convert async generator to SSE stream
        async def sse_stream():
            try:
                async for chunk in response:
                    if isinstance(chunk, BaseModel):
                        data = chunk.model_dump_json()
                    else:
                        data = json.dumps(chunk)
                    yield f"data: {data}\n\n"
            except Exception as e:
                # Send error as SSE event
                error_data = json.dumps({"error": {"message": str(e)}})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    return response


@router.post(
    "/embeddings",
    response_model=OpenAIEmbeddingsResponse,
    summary="Create embeddings.",
    description="Create embeddings.",
)
async def openai_embeddings(
    params: OpenAIEmbeddingsRequestWithExtraBody = Body(...),
    svc: InferenceService = Depends(get_inference_service),
) -> OpenAIEmbeddingsResponse:
    """Create embeddings."""
    return await svc.openai_embeddings(params=params)


@router.get(
    "/chat/completions",
    response_model=ListOpenAIChatCompletionResponse,
    summary="List chat completions.",
    description="List chat completions.",
)
async def list_chat_completions(
    after: str | None = Query(None, description="The ID of the last chat completion to return."),
    limit: int | None = Query(20, description="The maximum number of chat completions to return."),
    model: str | None = Query(None, description="The model to filter by."),
    order: Order | None = Query(
        Order.desc, description="The order to sort the chat completions by: 'asc' or 'desc'. Defaults to 'desc'."
    ),
    svc: InferenceService = Depends(get_inference_service),
) -> ListOpenAIChatCompletionResponse:
    """List chat completions."""
    return await svc.list_chat_completions(after=after, limit=limit, model=model, order=order)


@router.get(
    "/chat/completions/{completion_id}",
    response_model=OpenAICompletionWithInputMessages,
    summary="Get chat completion.",
    description="Get chat completion.",
)
async def get_chat_completion(
    completion_id: Annotated[str, FastAPIPath(..., description="ID of the chat completion.")],
    svc: InferenceService = Depends(get_inference_service),
) -> OpenAICompletionWithInputMessages:
    """Get chat completion."""
    return await svc.get_chat_completion(completion_id=completion_id)


# For backward compatibility with the router registry system
def create_inference_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Inference API (legacy compatibility)."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(router_v1alpha)
    return main_router


# Register the router factory
register_router(Api.inference, create_inference_router)
