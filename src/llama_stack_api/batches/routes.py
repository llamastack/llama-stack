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

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query

from llama_stack_api.batches import Batches, BatchObject, ListBatchesResponse
from llama_stack_api.batches.models import CreateBatchRequest, ListBatchesRequest
from llama_stack_api.datatypes import Api
from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1


def create_router(impl_getter: Callable[[Api], Batches]) -> APIRouter:
    """Create a FastAPI router for the Batches API.

    Args:
        impl_getter: Function that returns the Batches implementation for the batches API

    Returns:
        APIRouter configured for the Batches API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Batches"],
        responses=standard_responses,
    )

    def get_batch_service() -> Batches:
        """Dependency function to get the batch service implementation."""
        return impl_getter(Api.batches)

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
        svc: Annotated[Batches, Depends(get_batch_service)],
    ) -> BatchObject:
        return await svc.create_batch(request)

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
        batch_id: str,
        svc: Annotated[Batches, Depends(get_batch_service)],
    ) -> BatchObject:
        return await svc.retrieve_batch(batch_id)

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
        batch_id: str,
        svc: Annotated[Batches, Depends(get_batch_service)],
    ) -> BatchObject:
        return await svc.cancel_batch(batch_id)

    def get_list_batches_request(
        after: Annotated[
            str | None, Query(description="Optional cursor for pagination. Returns batches after this ID.")
        ] = None,
        limit: Annotated[int, Query(description="Maximum number of batches to return. Defaults to 20.")] = 20,
    ) -> ListBatchesRequest:
        """Dependency function to create ListBatchesRequest from query parameters."""
        return ListBatchesRequest(after=after, limit=limit)

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
        svc: Annotated[Batches, Depends(get_batch_service)],
    ) -> ListBatchesResponse:
        return await svc.list_batches(request)

    return router
