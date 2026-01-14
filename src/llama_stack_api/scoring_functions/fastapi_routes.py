# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the ScoringFunctions API.

This module defines the FastAPI router for the ScoringFunctions API using standard
FastAPI route decorators.

The router is defined in the API package to keep all API-related code together.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1

from .api import ScoringFunctions
from .models import (
    GetScoringFunctionRequest,
    ListScoringFunctionsRequest,
    ListScoringFunctionsResponse,
    RegisterScoringFunctionRequest,
    ScoringFn,
    UnregisterScoringFunctionRequest,
)

get_list_scoring_functions_request = create_query_dependency(ListScoringFunctionsRequest)
get_get_scoring_function_request = create_path_dependency(GetScoringFunctionRequest)
get_unregister_scoring_function_request = create_path_dependency(UnregisterScoringFunctionRequest)


def create_router(impl: ScoringFunctions) -> APIRouter:
    """Create a FastAPI router for the ScoringFunctions API.

    Args:
        impl: The ScoringFunctions implementation instance

    Returns:
        APIRouter configured for the ScoringFunctions API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Scoring Functions"],
        responses=standard_responses,
    )

    @router.get(
        "/scoring-functions",
        response_model=ListScoringFunctionsResponse,
        summary="List all scoring functions.",
        description="List all scoring functions.",
        responses={
            200: {"description": "A ListScoringFunctionsResponse."},
        },
    )
    async def list_scoring_functions(
        request: Annotated[ListScoringFunctionsRequest, Depends(get_list_scoring_functions_request)],
    ) -> ListScoringFunctionsResponse:
        return await impl.list_scoring_functions(request)

    @router.get(
        "/scoring-functions/{scoring_fn_id:path}",
        response_model=ScoringFn,
        summary="Get a scoring function by its ID.",
        description="Get a scoring function by its ID.",
        responses={
            200: {"description": "A ScoringFn."},
        },
    )
    async def get_scoring_function(
        request: Annotated[GetScoringFunctionRequest, Depends(get_get_scoring_function_request)],
    ) -> ScoringFn:
        return await impl.get_scoring_function(request)

    @router.post(
        "/scoring-functions",
        summary="Register a scoring function.",
        description="Register a scoring function.",
        responses={
            200: {"description": "The scoring function was successfully registered."},
        },
        deprecated=True,
    )
    async def register_scoring_function(
        request: Annotated[RegisterScoringFunctionRequest, Body(...)],
    ) -> None:
        return await impl.register_scoring_function(request)

    @router.delete(
        "/scoring-functions/{scoring_fn_id:path}",
        summary="Unregister a scoring function.",
        description="Unregister a scoring function.",
        responses={
            200: {"description": "The scoring function was successfully unregistered."},
        },
        deprecated=True,
    )
    async def unregister_scoring_function(
        request: Annotated[UnregisterScoringFunctionRequest, Depends(get_unregister_scoring_function_request)],
    ) -> None:
        return await impl.unregister_scoring_function(request)

    return router
