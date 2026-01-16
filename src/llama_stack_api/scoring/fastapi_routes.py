# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Scoring API.

This module defines the FastAPI router for the Scoring API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body

from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1

from .api import Scoring
from .models import ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse


def create_router(impl: Scoring) -> APIRouter:
    """Create a FastAPI router for the Scoring API.

    Args:
        impl: The Scoring implementation instance

    Returns:
        APIRouter configured for the Scoring API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Scoring"],
        responses=standard_responses,
    )

    @router.post(
        "/scoring/score",
        response_model=ScoreResponse,
        summary="Score a list of rows.",
        description="Score a list of rows.",
        responses={
            200: {"description": "A ScoreResponse object containing rows and aggregated results."},
        },
    )
    async def score(
        request: Annotated[ScoreRequest, Body(...)],
    ) -> ScoreResponse:
        return await impl.score(request)

    @router.post(
        "/scoring/score-batch",
        response_model=ScoreBatchResponse,
        summary="Score a batch of rows.",
        description="Score a batch of rows.",
        responses={
            200: {"description": "A ScoreBatchResponse."},
        },
    )
    async def score_batch(
        request: Annotated[ScoreBatchRequest, Body(...)],
    ) -> ScoreBatchResponse:
        return await impl.score_batch(request)

    return router
