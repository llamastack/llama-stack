# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1BETA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .datasetio_service import DatasetIOService


def get_datasetio_service(request: Request) -> DatasetIOService:
    """Dependency to get the datasetio service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.datasetio not in impls:
        raise ValueError("DatasetIO API implementation not found")
    return impls[Api.datasetio]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1BETA}",
    tags=["DatasetIO"],
    responses=standard_responses,
)


@router.get(
    "/datasetio/iterrows/{dataset_id:path}",
    response_model=PaginatedResponse,
    summary="Get a paginated list of rows from a dataset.",
    description="Get a paginated list of rows from a dataset using offset-based pagination.",
)
async def iterrows(
    dataset_id: Annotated[str, FastAPIPath(..., description="The ID of the dataset to get the rows from")],
    start_index: int | None = Query(
        None, description="Index into dataset for the first row to get. Get all rows if None."
    ),
    limit: int | None = Query(None, description="The number of rows to get."),
    svc: DatasetIOService = Depends(get_datasetio_service),
) -> PaginatedResponse:
    """Get a paginated list of rows from a dataset."""
    return await svc.iterrows(dataset_id=dataset_id, start_index=start_index, limit=limit)


@router.post(
    "/datasetio/append-rows/{dataset_id:path}",
    response_model=None,
    status_code=204,
    summary="Append rows to a dataset.",
    description="Append rows to a dataset.",
)
async def append_rows(
    dataset_id: Annotated[str, FastAPIPath(..., description="The ID of the dataset to append the rows to")],
    body: list[dict[str, Any]] = Body(..., description="The rows to append to the dataset."),
    svc: DatasetIOService = Depends(get_datasetio_service),
) -> None:
    """Append rows to a dataset."""
    await svc.append_rows(dataset_id=dataset_id, rows=body)


# For backward compatibility with the router registry system
def create_datasetio_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the DatasetIO API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.datasetio, create_datasetio_router)
