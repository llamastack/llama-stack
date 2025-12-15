# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Datasets API.

This module defines the FastAPI router for the Datasets API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack_api.router_utils import create_path_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1BETA

from .api import Datasets
from .models import (
    Dataset,
    GetDatasetRequest,
    ListDatasetsResponse,
    RegisterDatasetRequest,
    UnregisterDatasetRequest,
)

# Path parameter dependencies for single-field models
get_dataset_request = create_path_dependency(GetDatasetRequest)
unregister_dataset_request = create_path_dependency(UnregisterDatasetRequest)


def create_router(impl: Datasets) -> APIRouter:
    """Create a FastAPI router for the Datasets API.

    Args:
        impl: The Datasets implementation instance

    Returns:
        APIRouter configured for the Datasets API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1BETA}",
        tags=["Datasets"],
        responses=standard_responses,
    )

    @router.post(
        "/datasets",
        response_model=Dataset,
        summary="Register a new dataset.",
        description="Register a new dataset.",
        responses={
            200: {"description": "The registered dataset object."},
        },
        deprecated=True,
    )
    async def register_dataset(
        request: Annotated[RegisterDatasetRequest, Body(...)],
    ) -> Dataset:
        return await impl.register_dataset(request)

    @router.get(
        "/datasets/{dataset_id:path}",
        response_model=Dataset,
        summary="Get a dataset by its ID.",
        description="Get a dataset by its ID.",
        responses={
            200: {"description": "The dataset object."},
        },
    )
    async def get_dataset(
        request: Annotated[GetDatasetRequest, Depends(get_dataset_request)],
    ) -> Dataset:
        return await impl.get_dataset(request)

    @router.get(
        "/datasets",
        response_model=ListDatasetsResponse,
        summary="List all datasets.",
        description="List all datasets.",
        responses={
            200: {"description": "A list of dataset objects."},
        },
    )
    async def list_datasets() -> ListDatasetsResponse:
        return await impl.list_datasets()

    @router.delete(
        "/datasets/{dataset_id:path}",
        summary="Unregister a dataset by its ID.",
        description="Unregister a dataset by its ID.",
        responses={
            200: {"description": "The dataset was successfully unregistered."},
        },
        deprecated=True,
    )
    async def unregister_dataset(
        request: Annotated[UnregisterDatasetRequest, Depends(unregister_dataset_request)],
    ) -> None:
        return await impl.unregister_dataset(request)

    return router
