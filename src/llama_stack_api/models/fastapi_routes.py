# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Models API.

This module defines the FastAPI router for the Models API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack_api.router_utils import create_path_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1

from .api import Models
from .models import (
    GetModelRequest,
    Model,
    OpenAIListModelsResponse,
    RegisterModelRequest,
    UnregisterModelRequest,
)

# Path parameter dependencies for single-field models
get_model_request = create_path_dependency(GetModelRequest)
unregister_model_request = create_path_dependency(UnregisterModelRequest)


def create_router(impl: Models) -> APIRouter:
    """Create a FastAPI router for the Models API.

    Args:
        impl: The Models implementation instance

    Returns:
        APIRouter configured for the Models API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Models"],
        responses=standard_responses,
    )

    @router.get(
        "/models",
        response_model=OpenAIListModelsResponse,
        summary="List models using the OpenAI API.",
        description="List models using the OpenAI API.",
        responses={
            200: {"description": "A list of OpenAI model objects."},
        },
    )
    async def openai_list_models() -> OpenAIListModelsResponse:
        return await impl.openai_list_models()

    @router.get(
        "/models/{model_id:path}",
        response_model=Model,
        summary="Get a model by its identifier.",
        description="Get a model by its identifier.",
        responses={
            200: {"description": "The model object."},
        },
    )
    async def get_model(
        request: Annotated[GetModelRequest, Depends(get_model_request)],
    ) -> Model:
        return await impl.get_model(request)

    @router.post(
        "/models",
        response_model=Model,
        summary="Register a model.",
        description="Register a model.",
        responses={
            200: {"description": "The registered model object."},
        },
        deprecated=True,
    )
    async def register_model(
        request: Annotated[RegisterModelRequest, Body(...)],
    ) -> Model:
        return await impl.register_model(request)

    @router.delete(
        "/models/{model_id:path}",
        summary="Unregister a model.",
        description="Unregister a model.",
        responses={
            200: {"description": "The model was successfully unregistered."},
        },
        deprecated=True,
    )
    async def unregister_model(
        request: Annotated[UnregisterModelRequest, Depends(unregister_model_request)],
    ) -> None:
        return await impl.unregister_model(request)

    return router
