# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .model_schemas import (
    ListModelsResponse,
    Model,
    RegisterModelRequest,
)
from .models_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Dependency to get the model service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.models not in impls:
        raise ValueError("Models API implementation not found")
    return impls[Api.models]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Models"],
    responses=standard_responses,
)


@router.get(
    "/models",
    response_model=ListModelsResponse,
    summary="List all models.",
    description="List all models registered in Llama Stack.",
)
async def list_models(svc: ModelService = Depends(get_model_service)) -> ListModelsResponse:
    """List all models."""
    return await svc.list_models()


@router.get(
    "/models/{model_id:path}",
    response_model=Model,
    summary="Get model.",
    description="Get a model by its identifier.",
)
async def get_model(
    model_id: Annotated[str, FastAPIPath(..., description="The identifier of the model to get.")],
    svc: ModelService = Depends(get_model_service),
) -> Model:
    """Get model by its identifier."""
    return await svc.get_model(model_id=model_id)


@router.post(
    "/models",
    response_model=Model,
    summary="Register model.",
    description="Register a new model in Llama Stack.",
)
async def register_model(
    body: RegisterModelRequest = Body(...),
    svc: ModelService = Depends(get_model_service),
) -> Model:
    """Register a new model."""
    return await svc.register_model(
        model_id=body.model_id,
        provider_model_id=body.provider_model_id,
        provider_id=body.provider_id,
        metadata=body.metadata,
        model_type=body.model_type,
    )


@router.delete(
    "/models/{model_id:path}",
    response_model=None,
    status_code=204,
    summary="Unregister model.",
    description="Unregister a model from Llama Stack.",
)
async def unregister_model(
    model_id: Annotated[str, FastAPIPath(..., description="The identifier of the model to unregister.")],
    svc: ModelService = Depends(get_model_service),
) -> None:
    """Unregister a model."""
    await svc.unregister_model(model_id=model_id)


# For backward compatibility with the router registry system
def create_models_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Models API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.models, create_models_router)
