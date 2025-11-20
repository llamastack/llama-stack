# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Batches API.

This module defines the FastAPI router for the Batches API using standard
FastAPI route decorators instead of Protocol-based route discovery.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack.core.server.router_registry import register_router
from llama_stack.core.server.router_utils import standard_responses
from llama_stack_api.batches import Batches, BatchObject, ListBatchesResponse
from llama_stack_api.batches.models import CreateBatchRequest
from llama_stack_api.datatypes import Api
from llama_stack_api.version import LLAMA_STACK_API_V1


def create_batches_router(impl_getter: Callable[[Api], Batches]) -> APIRouter:
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
        svc: Annotated[Batches, Depends(get_batch_service)],
        after: str | None = None,
        limit: int = 20,
    ) -> ListBatchesResponse:
        return await svc.list_batches(after=after, limit=limit)

    return router


# Register the router factory
register_router(Api.batches, create_batches_router)
