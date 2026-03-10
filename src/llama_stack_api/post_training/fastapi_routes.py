# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Post-Training API.

This module defines the FastAPI router for the Post-Training API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack_api.router_utils import create_path_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA

from .api import PostTraining
from .models import (
    CancelTrainingJobRequest,
    GetTrainingJobArtifactsRequest,
    GetTrainingJobStatusRequest,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    PreferenceOptimizeRequest,
    SupervisedFineTuneRequest,
)

# Path parameter dependencies for single-field models
get_training_job_status_request = create_path_dependency(GetTrainingJobStatusRequest)
cancel_training_job_request = create_path_dependency(CancelTrainingJobRequest)
get_training_job_artifacts_request = create_path_dependency(GetTrainingJobArtifactsRequest)


def create_router(impl: PostTraining) -> APIRouter:
    """Create a FastAPI router for the Post-Training API."""
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
        tags=["Post Training"],
        responses=standard_responses,
    )

    @router.post(
        "/post-training/supervised-fine-tune",
        response_model=PostTrainingJob,
        summary="Run supervised fine-tuning of a model.",
        description="Run supervised fine-tuning of a model.",
        responses={200: {"description": "A PostTrainingJob."}},
    )
    async def supervised_fine_tune(
        request: Annotated[SupervisedFineTuneRequest, Body(...)],
    ) -> PostTrainingJob:
        return await impl.supervised_fine_tune(request)

    @router.post(
        "/post-training/preference-optimize",
        response_model=PostTrainingJob,
        summary="Run preference optimization of a model.",
        description="Run preference optimization of a model.",
        responses={200: {"description": "A PostTrainingJob."}},
    )
    async def preference_optimize(
        request: Annotated[PreferenceOptimizeRequest, Body(...)],
    ) -> PostTrainingJob:
        return await impl.preference_optimize(request)

    @router.get(
        "/post-training/jobs",
        response_model=ListPostTrainingJobsResponse,
        summary="Get all training jobs.",
        description="Get all training jobs.",
        responses={200: {"description": "A ListPostTrainingJobsResponse."}},
    )
    async def get_training_jobs() -> ListPostTrainingJobsResponse:
        return await impl.get_training_jobs()

    @router.get(
        "/post-training/jobs/{job_uuid}/status",
        response_model=PostTrainingJobStatusResponse,
        summary="Get the status of a training job.",
        description="Get the status of a training job.",
        responses={200: {"description": "A PostTrainingJobStatusResponse."}},
    )
    async def get_training_job_status(
        request: Annotated[GetTrainingJobStatusRequest, Depends(get_training_job_status_request)],
    ) -> PostTrainingJobStatusResponse:
        return await impl.get_training_job_status(request)

    @router.post(
        "/post-training/jobs/{job_uuid}/cancel",
        summary="Cancel a training job.",
        description="Cancel a training job.",
        responses={200: {"description": "Successfully cancelled the training job."}},
    )
    async def cancel_training_job(
        request: Annotated[CancelTrainingJobRequest, Depends(cancel_training_job_request)],
    ) -> None:
        return await impl.cancel_training_job(request)

    @router.get(
        "/post-training/jobs/{job_uuid}/artifacts",
        response_model=PostTrainingJobArtifactsResponse,
        summary="Get the artifacts of a training job.",
        description="Get the artifacts of a training job.",
        responses={200: {"description": "A PostTrainingJobArtifactsResponse."}},
    )
    async def get_training_job_artifacts(
        request: Annotated[GetTrainingJobArtifactsRequest, Depends(get_training_job_artifacts_request)],
    ) -> PostTrainingJobArtifactsResponse:
        return await impl.get_training_job_artifacts(request)

    return router
