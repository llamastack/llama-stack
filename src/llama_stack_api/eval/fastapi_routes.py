# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack_api.common.job_types import Job
from llama_stack_api.router_utils import create_path_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA

from .api import Eval
from .models import (
    BenchmarkIdRequest,
    EvaluateResponse,
    EvaluateRowsBodyRequest,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    RunEvalBodyRequest,
    RunEvalRequest,
)

get_benchmark_id_request = create_path_dependency(BenchmarkIdRequest)


def create_router(impl: Eval) -> APIRouter:
    """Create a FastAPI router for the Eval API."""
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
        tags=["Eval"],
        responses=standard_responses,
    )

    @router.post(
        "/eval/benchmarks/{benchmark_id}/jobs",
        response_model=Job,
        summary="Run Eval",
        description="Run an evaluation on a benchmark.",
        responses={
            200: {"description": "The job that was created to run the evaluation."},
        },
    )
    async def run_eval(
        benchmark_id_request: Annotated[BenchmarkIdRequest, Depends(get_benchmark_id_request)],
        body_request: Annotated[RunEvalBodyRequest, Body(...)],
    ) -> Job:
        request = RunEvalRequest(
            benchmark_id=benchmark_id_request.benchmark_id,
            benchmark_config=body_request.benchmark_config,
        )
        return await impl.run_eval(request)

    @router.post(
        "/eval/benchmarks/{benchmark_id}/evaluations",
        response_model=EvaluateResponse,
        summary="Evaluate Rows",
        description="Evaluate a list of rows on a benchmark.",
        responses={
            200: {"description": "EvaluateResponse object containing generations and scores."},
        },
    )
    async def evaluate_rows(
        benchmark_id_request: Annotated[BenchmarkIdRequest, Depends(get_benchmark_id_request)],
        body_request: Annotated[EvaluateRowsBodyRequest, Body(...)],
    ) -> EvaluateResponse:
        request = EvaluateRowsRequest(
            benchmark_id=benchmark_id_request.benchmark_id,
            input_rows=body_request.input_rows,
            scoring_functions=body_request.scoring_functions,
            benchmark_config=body_request.benchmark_config,
        )
        return await impl.evaluate_rows(request)

    @router.get(
        "/eval/benchmarks/{benchmark_id}/jobs/{job_id}",
        response_model=Job,
        summary="Job Status",
        description="Get the status of a job.",
        responses={
            200: {"description": "The status of the evaluation job."},
        },
    )
    async def job_status(
        benchmark_id: str,
        job_id: str,
    ) -> Job:
        request = JobStatusRequest(benchmark_id=benchmark_id, job_id=job_id)
        return await impl.job_status(request)

    @router.delete(
        "/eval/benchmarks/{benchmark_id}/jobs/{job_id}",
        summary="Job Cancel",
        description="Cancel a job.",
        responses={
            200: {"description": "Successful Response"},
        },
    )
    async def job_cancel(
        benchmark_id: str,
        job_id: str,
    ) -> None:
        request = JobCancelRequest(benchmark_id=benchmark_id, job_id=job_id)
        return await impl.job_cancel(request)

    @router.get(
        "/eval/benchmarks/{benchmark_id}/jobs/{job_id}/result",
        response_model=EvaluateResponse,
        summary="Job Result",
        description="Get the result of a job.",
        responses={
            200: {"description": "The result of the job."},
        },
    )
    async def job_result(
        benchmark_id: str,
        job_id: str,
    ) -> EvaluateResponse:
        request = JobResultRequest(benchmark_id=benchmark_id, job_id=job_id)
        return await impl.job_result(request)

    return router
