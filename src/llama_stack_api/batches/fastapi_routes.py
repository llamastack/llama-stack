# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Batches API.

This module defines the FastAPI router for the Batches API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query

from llama_stack_api.batches import Batches, BatchObject, ListBatchesResponse
from llama_stack_api.batches.models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)
from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1


def get_retrieve_batch_request(
    batch_id: Annotated[str, Path(description="The ID of the batch to retrieve.")],
) -> RetrieveBatchRequest:
    """Dependency function to create RetrieveBatchRequest from path parameter."""
    return RetrieveBatchRequest(batch_id=batch_id)


def get_cancel_batch_request(
    batch_id: Annotated[str, Path(description="The ID of the batch to cancel.")],
) -> CancelBatchRequest:
    """Dependency function to create CancelBatchRequest from path parameter."""
    return CancelBatchRequest(batch_id=batch_id)


def get_list_batches_request(
    after: Annotated[
        str | None, Query(description="Optional cursor for pagination. Returns batches after this ID.")
    ] = None,
    limit: Annotated[int, Query(description="Maximum number of batches to return. Defaults to 20.")] = 20,
) -> ListBatchesRequest:
    """Dependency function to create ListBatchesRequest from query parameters."""
    return ListBatchesRequest(after=after, limit=limit)


def create_router(impl: Batches) -> APIRouter:
    """Create a FastAPI router for the Batches API.

    Args:
        impl: The Batches implementation instance

    Returns:
        APIRouter configured for the Batches API
    """
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
        responses={
            200: {"description": "The created batch object."},
            409: {"description": "Conflict: The idempotency key was previously used with different parameters."},
        },
    )
    async def create_batch(
        request: Annotated[CreateBatchRequest, Body(...)],
    ) -> BatchObject:
        return await impl.create_batch(request)

    @router.get(
        "/batches/{batch_id}",
        response_model=BatchObject,
        summary="Retrieve information about a specific batch.",
        description="Retrieve information about a specific batch.",
        responses={
            200: {"description": "The batch object."},
        },
    )
    async def retrieve_batch(
        request: Annotated[RetrieveBatchRequest, Depends(get_retrieve_batch_request)],
    ) -> BatchObject:
        return await impl.retrieve_batch(request)

    @router.post(
        "/batches/{batch_id}/cancel",
        response_model=BatchObject,
        summary="Cancel a batch that is in progress.",
        description="Cancel a batch that is in progress.",
        responses={
            200: {"description": "The updated batch object."},
        },
    )
    async def cancel_batch(
        request: Annotated[CancelBatchRequest, Depends(get_cancel_batch_request)],
    ) -> BatchObject:
        return await impl.cancel_batch(request)

    @router.get(
        "/batches",
        response_model=ListBatchesResponse,
        summary="List all batches for the current user.",
        description="List all batches for the current user.",
        responses={
            200: {"description": "A list of batch objects."},
        },
    )
    async def list_batches(
        request: Annotated[ListBatchesRequest, Depends(get_list_batches_request)],
    ) -> ListBatchesResponse:
        return await impl.list_batches(request)

    return router
