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

from .models import (
    ListScoringFunctionsResponse,
    RegisterScoringFunctionRequest,
    ScoringFn,
)
from .scoring_functions_service import ScoringFunctionsService


def get_scoring_functions_service(request: Request) -> ScoringFunctionsService:
    """Dependency to get the scoring functions service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.scoring_functions not in impls:
        raise ValueError("Scoring Functions API implementation not found")
    return impls[Api.scoring_functions]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Scoring Functions"],
    responses=standard_responses,
)


@router.get(
    "/scoring-functions",
    response_model=ListScoringFunctionsResponse,
    summary="List all scoring functions",
    description="List all scoring functions",
)
async def list_scoring_functions(
    svc: ScoringFunctionsService = Depends(get_scoring_functions_service),
) -> ListScoringFunctionsResponse:
    """List all scoring functions."""
    return await svc.list_scoring_functions()


@router.get(
    "/scoring-functions/{scoring_fn_id:path}",
    response_model=ScoringFn,
    summary="Get a scoring function by its ID",
    description="Get a scoring function by its ID",
)
async def get_scoring_function(
    scoring_fn_id: Annotated[str, FastAPIPath(..., description="The ID of the scoring function to get")],
    svc: ScoringFunctionsService = Depends(get_scoring_functions_service),
) -> ScoringFn:
    """Get a scoring function by its ID."""
    return await svc.get_scoring_function(scoring_fn_id)


@router.post(
    "/scoring-functions",
    response_model=None,
    status_code=204,
    summary="Register a scoring function",
    description="Register a scoring function",
)
async def register_scoring_function(
    body: RegisterScoringFunctionRequest = Body(...),
    svc: ScoringFunctionsService = Depends(get_scoring_functions_service),
) -> None:
    """Register a scoring function."""
    return await svc.register_scoring_function(
        scoring_fn_id=body.scoring_fn_id,
        description=body.description,
        return_type=body.return_type,
        provider_scoring_fn_id=body.provider_scoring_fn_id,
        provider_id=body.provider_id,
        params=body.params,
    )


@router.delete(
    "/scoring-functions/{scoring_fn_id:path}",
    response_model=None,
    status_code=204,
    summary="Unregister a scoring function",
    description="Unregister a scoring function",
)
async def unregister_scoring_function(
    scoring_fn_id: Annotated[str, FastAPIPath(..., description="The ID of the scoring function to unregister")],
    svc: ScoringFunctionsService = Depends(get_scoring_functions_service),
) -> None:
    """Unregister a scoring function."""
    await svc.unregister_scoring_function(scoring_fn_id)


# For backward compatibility with the router registry system
def create_scoring_functions_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Scoring Functions API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.scoring_functions, create_scoring_functions_router)
