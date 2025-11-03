# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1BETA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .datasets_service import DatasetsService
from .models import Dataset, ListDatasetsResponse, RegisterDatasetRequest


def get_datasets_service(request: Request) -> DatasetsService:
    """Dependency to get the datasets service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.datasets not in impls:
        raise ValueError("Datasets API implementation not found")
    return impls[Api.datasets]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Datasets"],
    responses=standard_responses,
)

router_v1beta = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1BETA}",
    tags=["Datasets"],
    responses=standard_responses,
)


@router.post(
    "/datasets",
    response_model=Dataset,
    summary="Register a new dataset",
    description="Register a new dataset",
    deprecated=True,
)
@router_v1beta.post(
    "/datasets",
    response_model=Dataset,
    summary="Register a new dataset",
    description="Register a new dataset",
)
async def register_dataset(
    body: RegisterDatasetRequest = Body(...),
    svc: DatasetsService = Depends(get_datasets_service),
) -> Dataset:
    """Register a new dataset."""
    return await svc.register_dataset(
        purpose=body.purpose,
        source=body.source,
        metadata=body.metadata,
        dataset_id=body.dataset_id,
    )


@router.get(
    "/datasets/{dataset_id:path}",
    response_model=Dataset,
    summary="Get a dataset by its ID",
    description="Get a dataset by its ID",
    deprecated=True,
)
@router_v1beta.get(
    "/datasets/{{dataset_id:path}}",
    response_model=Dataset,
    summary="Get a dataset by its ID",
    description="Get a dataset by its ID",
)
async def get_dataset(
    dataset_id: Annotated[str, FastAPIPath(..., description="The ID of the dataset to get")],
    svc: DatasetsService = Depends(get_datasets_service),
) -> Dataset:
    """Get a dataset by its ID."""
    return await svc.get_dataset(dataset_id=dataset_id)


@router.get(
    "/datasets",
    response_model=ListDatasetsResponse,
    summary="List all datasets",
    description="List all datasets",
    deprecated=True,
)
@router_v1beta.get(
    "/datasets",
    response_model=ListDatasetsResponse,
    summary="List all datasets",
    description="List all datasets",
)
async def list_datasets(svc: DatasetsService = Depends(get_datasets_service)) -> ListDatasetsResponse:
    """List all datasets."""
    return await svc.list_datasets()


@router.delete(
    "/datasets/{dataset_id:path}",
    response_model=None,
    status_code=204,
    summary="Unregister a dataset by its ID",
    description="Unregister a dataset by its ID",
    deprecated=True,
)
@router_v1beta.delete(
    "/datasets/{{dataset_id:path}}",
    response_model=None,
    status_code=204,
    summary="Unregister a dataset by its ID",
    description="Unregister a dataset by its ID",
)
async def unregister_dataset(
    dataset_id: Annotated[str, FastAPIPath(..., description="The ID of the dataset to unregister")],
    svc: DatasetsService = Depends(get_datasets_service),
) -> None:
    """Unregister a dataset by its ID."""
    await svc.unregister_dataset(dataset_id=dataset_id)


# For backward compatibility with the router registry system
def create_datasets_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Datasets API (legacy compatibility)."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(router_v1beta)
    return main_router


# Register the router factory
register_router(Api.datasets, create_datasets_router)
