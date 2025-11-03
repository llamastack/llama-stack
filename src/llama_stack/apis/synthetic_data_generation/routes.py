# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from fastapi import Body, Depends, Request

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import (
    SyntheticDataGenerationRequest,
    SyntheticDataGenerationResponse,
)
from .synthetic_data_generation_service import SyntheticDataGenerationService


def get_synthetic_data_generation_service(request: Request) -> SyntheticDataGenerationService:
    """Dependency to get the synthetic data generation service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.synthetic_data_generation not in impls:
        raise ValueError("Synthetic Data Generation API implementation not found")
    return impls[Api.synthetic_data_generation]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Synthetic Data Generation"],
    responses=standard_responses,
)


@router.post(
    "/synthetic-data-generation/generate",
    response_model=SyntheticDataGenerationResponse,
    summary="Generate synthetic data based on input dialogs and apply filtering",
    description="Generate synthetic data based on input dialogs and apply filtering",
)
def synthetic_data_generate(
    body: SyntheticDataGenerationRequest = Body(...),
    svc: SyntheticDataGenerationService = Depends(get_synthetic_data_generation_service),
) -> SyntheticDataGenerationResponse:
    """Generate synthetic data based on input dialogs and apply filtering."""
    return svc.synthetic_data_generate(
        dialogs=body.dialogs,
        filtering_function=body.filtering_function,
        model=body.model,
    )


# For backward compatibility with the router registry system
def create_synthetic_data_generation_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Synthetic Data Generation API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.synthetic_data_generation, create_synthetic_data_generation_router)
