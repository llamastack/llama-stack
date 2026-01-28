# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Fine-Tuning API.

This module defines FastAPI routes matching OpenAI's fine-tuning API structure.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Path, Query

from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1

from .api import FineTuning
from .models import (
    CreateFineTuningJobRequest,
    FineTuningJob,
    ListFineTuningJobCheckpointsResponse,
    ListFineTuningJobEventsResponse,
    ListFineTuningJobsResponse,
)


def create_router(impl: FineTuning) -> APIRouter:
    """Create FastAPI router for fine-tuning API."""
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Fine-tuning"],
        responses=standard_responses,
    )

    @router.post(
        "/fine_tuning/jobs",
        response_model=FineTuningJob,
        summary="Create fine-tuning job",
        description="Creates a fine-tuning job for a specified model and dataset.",
        responses={200: {"description": "Fine-tuning job created successfully"}},
    )
    async def create_fine_tuning_job(
        request: Annotated[CreateFineTuningJobRequest, Body(...)],
    ) -> FineTuningJob:
        return await impl.create_fine_tuning_job(request)

    @router.get(
        "/fine_tuning/jobs",
        response_model=ListFineTuningJobsResponse,
        summary="List fine-tuning jobs",
        description="List all fine-tuning jobs with pagination support.",
        responses={200: {"description": "List of fine-tuning jobs"}},
    )
    async def list_fine_tuning_jobs(
        after: str | None = None,
        limit: Annotated[int, Query(description="Number of results", le=100)] = 20,
    ) -> ListFineTuningJobsResponse:
        return await impl.list_fine_tuning_jobs(after=after, limit=limit)

    @router.get(
        "/fine_tuning/jobs/{fine_tuning_job_id}",
        response_model=FineTuningJob,
        summary="Retrieve fine-tuning job",
        description="Get detailed information about a specific fine-tuning job.",
        responses={200: {"description": "Fine-tuning job details"}},
    )
    async def retrieve_fine_tuning_job(
        fine_tuning_job_id: Annotated[str, Path(description="Fine-tuning job ID")],
    ) -> FineTuningJob:
        return await impl.retrieve_fine_tuning_job(fine_tuning_job_id)

    @router.post(
        "/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
        response_model=FineTuningJob,
        summary="Cancel fine-tuning job",
        description="Immediately cancels a running fine-tuning job.",
        responses={200: {"description": "Fine-tuning job cancelled"}},
    )
    async def cancel_fine_tuning_job(
        fine_tuning_job_id: Annotated[str, Path(description="Fine-tuning job ID")],
    ) -> FineTuningJob:
        return await impl.cancel_fine_tuning_job(fine_tuning_job_id)

    @router.get(
        "/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints",
        response_model=ListFineTuningJobCheckpointsResponse,
        summary="List job checkpoints",
        description="List checkpoints created during fine-tuning.",
        responses={200: {"description": "List of checkpoints"}},
    )
    async def list_fine_tuning_job_checkpoints(
        fine_tuning_job_id: Annotated[str, Path(description="Fine-tuning job ID")],
        after: str | None = None,
        limit: Annotated[int, Query(description="Number of results", le=100)] = 10,
    ) -> ListFineTuningJobCheckpointsResponse:
        return await impl.list_fine_tuning_job_checkpoints(fine_tuning_job_id, after=after, limit=limit)

    @router.get(
        "/fine_tuning/jobs/{fine_tuning_job_id}/events",
        response_model=ListFineTuningJobEventsResponse,
        summary="List job events",
        description="List events and logs from a fine-tuning job.",
        responses={200: {"description": "List of events"}},
    )
    async def list_fine_tuning_events(
        fine_tuning_job_id: Annotated[str, Path(description="Fine-tuning job ID")],
        after: str | None = None,
        limit: Annotated[int, Query(description="Number of results", le=100)] = 20,
    ) -> ListFineTuningJobEventsResponse:
        return await impl.list_fine_tuning_events(fine_tuning_job_id, after=after, limit=limit)

    return router
