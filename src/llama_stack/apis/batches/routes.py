# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

try:
    from openai.types import Batch as BatchObject
except ImportError as e:
    raise ImportError("OpenAI package is required for batches API. Please install it with: pip install openai") from e

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .batches_service import BatchService
from .models import CreateBatchRequest, ListBatchesResponse


def get_batch_service(request: Request) -> BatchService:
    """Dependency to get the batch service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.batches not in impls:
        raise ValueError("Batches API implementation not found")
    return impls[Api.batches]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Batches"],
    responses=standard_responses,
)


@router.post(
    "/batches",
    response_model=BatchObject,
    summary="Create a new batch for processing multiple API requests.",
    description="Create a new batch for processing multiple API requests.",
)
async def create_batch(
    request: CreateBatchRequest = Body(...),
    svc: BatchService = Depends(get_batch_service),
) -> BatchObject:
    """Create a new batch."""
    return await svc.create_batch(
        input_file_id=request.input_file_id,
        endpoint=request.endpoint,
        completion_window=request.completion_window,
        metadata=request.metadata,
        idempotency_key=request.idempotency_key,
    )


@router.get(
    "/batches/{batch_id}",
    response_model=BatchObject,
    summary="Retrieve information about a specific batch.",
    description="Retrieve information about a specific batch.",
)
async def retrieve_batch(
    batch_id: Annotated[str, FastAPIPath(..., description="The ID of the batch to retrieve.")],
    svc: BatchService = Depends(get_batch_service),
) -> BatchObject:
    """Retrieve batch information."""
    return await svc.retrieve_batch(batch_id)


@router.post(
    "/batches/{batch_id}/cancel",
    response_model=BatchObject,
    summary="Cancel a batch that is in progress.",
    description="Cancel a batch that is in progress.",
)
async def cancel_batch(
    batch_id: Annotated[str, FastAPIPath(..., description="The ID of the batch to cancel.")],
    svc: BatchService = Depends(get_batch_service),
) -> BatchObject:
    """Cancel a batch."""
    return await svc.cancel_batch(batch_id)


@router.get(
    "/batches",
    response_model=ListBatchesResponse,
    summary="List all batches for the current user.",
    description="List all batches for the current user.",
)
async def list_batches(
    after: str | None = Query(None, description="A cursor for pagination; returns batches after this batch ID."),
    limit: int = Query(20, description="Number of batches to return (default 20, max 100).", ge=1, le=100),
    svc: BatchService = Depends(get_batch_service),
) -> ListBatchesResponse:
    """List all batches."""
    return await svc.list_batches(after=after, limit=limit)


# For backward compatibility with the router registry system
def create_batches_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Batches API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.batches, create_batches_router)
