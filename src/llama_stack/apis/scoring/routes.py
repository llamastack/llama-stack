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

from .models import ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse
from .scoring_service import ScoringService


def get_scoring_service(request: Request) -> ScoringService:
    """Dependency to get the scoring service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.scoring not in impls:
        raise ValueError("Scoring API implementation not found")
    return impls[Api.scoring]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Scoring"],
    responses=standard_responses,
)


@router.post(
    "/scoring/score-batch",
    response_model=ScoreBatchResponse,
    summary="Score a batch of rows",
    description="Score a batch of rows from a dataset",
)
async def score_batch(
    body: ScoreBatchRequest = Body(...),
    svc: ScoringService = Depends(get_scoring_service),
) -> ScoreBatchResponse:
    """Score a batch of rows from a dataset."""
    return await svc.score_batch(
        dataset_id=body.dataset_id,
        scoring_functions=body.scoring_functions,
        save_results_dataset=body.save_results_dataset,
    )


@router.post(
    "/scoring/score",
    response_model=ScoreResponse,
    summary="Score a list of rows",
    description="Score a list of rows",
)
async def score(
    body: ScoreRequest = Body(...),
    svc: ScoringService = Depends(get_scoring_service),
) -> ScoreResponse:
    """Score a list of rows."""
    return await svc.score(
        input_rows=body.input_rows,
        scoring_functions=body.scoring_functions,
    )


# For backward compatibility with the router registry system
def create_scoring_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Scoring API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.scoring, create_scoring_router)
