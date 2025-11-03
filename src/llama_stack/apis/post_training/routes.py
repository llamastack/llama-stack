# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from fastapi import Body, Depends, Query, Request

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import (
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    PreferenceOptimizeRequest,
    SupervisedFineTuneRequest,
)
from .post_training_service import PostTrainingService


def get_post_training_service(request: Request) -> PostTrainingService:
    """Dependency to get the post training service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.post_training not in impls:
        raise ValueError("Post Training API implementation not found")
    return impls[Api.post_training]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Post Training"],
    responses=standard_responses,
)

router_v1alpha = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
    tags=["Post Training"],
    responses=standard_responses,
)


@router.post(
    "/post-training/supervised-fine-tune",
    response_model=PostTrainingJob,
    summary="Run supervised fine-tuning of a model",
    description="Run supervised fine-tuning of a model",
    deprecated=True,
)
@router_v1alpha.post(
    "/post-training/supervised-fine-tune",
    response_model=PostTrainingJob,
    summary="Run supervised fine-tuning of a model",
    description="Run supervised fine-tuning of a model",
)
async def supervised_fine_tune(
    body: SupervisedFineTuneRequest = Body(...),
    svc: PostTrainingService = Depends(get_post_training_service),
) -> PostTrainingJob:
    """Run supervised fine-tuning of a model."""
    return await svc.supervised_fine_tune(
        job_uuid=body.job_uuid,
        training_config=body.training_config,
        hyperparam_search_config=body.hyperparam_search_config,
        logger_config=body.logger_config,
        model=body.model,
        checkpoint_dir=body.checkpoint_dir,
        algorithm_config=body.algorithm_config,
    )


@router.post(
    "/post-training/preference-optimize",
    response_model=PostTrainingJob,
    summary="Run preference optimization of a model",
    description="Run preference optimization of a model",
    deprecated=True,
)
@router_v1alpha.post(
    "/post-training/preference-optimize",
    response_model=PostTrainingJob,
    summary="Run preference optimization of a model",
    description="Run preference optimization of a model",
)
async def preference_optimize(
    body: PreferenceOptimizeRequest = Body(...),
    svc: PostTrainingService = Depends(get_post_training_service),
) -> PostTrainingJob:
    """Run preference optimization of a model."""
    return await svc.preference_optimize(
        job_uuid=body.job_uuid,
        finetuned_model=body.finetuned_model,
        algorithm_config=body.algorithm_config,
        training_config=body.training_config,
        hyperparam_search_config=body.hyperparam_search_config,
        logger_config=body.logger_config,
    )


@router.get(
    "/post-training/jobs",
    response_model=ListPostTrainingJobsResponse,
    summary="Get all training jobs",
    description="Get all training jobs",
    deprecated=True,
)
@router_v1alpha.get(
    "/post-training/jobs",
    response_model=ListPostTrainingJobsResponse,
    summary="Get all training jobs",
    description="Get all training jobs",
)
async def get_training_jobs(
    svc: PostTrainingService = Depends(get_post_training_service),
) -> ListPostTrainingJobsResponse:
    """Get all training jobs."""
    return await svc.get_training_jobs()


@router.get(
    "/post-training/job/status",
    response_model=PostTrainingJobStatusResponse,
    summary="Get the status of a training job",
    description="Get the status of a training job",
    deprecated=True,
)
@router_v1alpha.get(
    "/post-training/job/status",
    response_model=PostTrainingJobStatusResponse,
    summary="Get the status of a training job",
    description="Get the status of a training job",
)
async def get_training_job_status(
    job_uuid: str = Query(..., description="The UUID of the job to get the status of"),
    svc: PostTrainingService = Depends(get_post_training_service),
) -> PostTrainingJobStatusResponse:
    """Get the status of a training job."""
    return await svc.get_training_job_status(job_uuid=job_uuid)


@router.post(
    "/post-training/job/cancel",
    response_model=None,
    status_code=204,
    summary="Cancel a training job",
    description="Cancel a training job",
    deprecated=True,
)
@router_v1alpha.post(
    "/post-training/job/cancel",
    response_model=None,
    status_code=204,
    summary="Cancel a training job",
    description="Cancel a training job",
)
async def cancel_training_job(
    job_uuid: str = Query(..., description="The UUID of the job to cancel"),
    svc: PostTrainingService = Depends(get_post_training_service),
) -> None:
    """Cancel a training job."""
    await svc.cancel_training_job(job_uuid=job_uuid)


@router.get(
    "/post-training/job/artifacts",
    response_model=PostTrainingJobArtifactsResponse,
    summary="Get the artifacts of a training job",
    description="Get the artifacts of a training job",
    deprecated=True,
)
@router_v1alpha.get(
    "/post-training/job/artifacts",
    response_model=PostTrainingJobArtifactsResponse,
    summary="Get the artifacts of a training job",
    description="Get the artifacts of a training job",
)
async def get_training_job_artifacts(
    job_uuid: str = Query(..., description="The UUID of the job to get the artifacts of"),
    svc: PostTrainingService = Depends(get_post_training_service),
) -> PostTrainingJobArtifactsResponse:
    """Get the artifacts of a training job."""
    return await svc.get_training_job_artifacts(job_uuid=job_uuid)


# For backward compatibility with the router registry system
def create_post_training_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Post Training API (legacy compatibility)."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(router_v1alpha)
    return main_router


# Register the router factory
register_router(Api.post_training, create_post_training_router)
