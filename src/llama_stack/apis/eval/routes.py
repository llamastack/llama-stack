# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.common.job_types import Job
from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .eval_service import EvalService
from .models import BenchmarkConfig, EvaluateResponse, EvaluateRowsRequest


def get_eval_service(request: Request) -> EvalService:
    """Dependency to get the eval service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.eval not in impls:
        raise ValueError("Eval API implementation not found")
    return impls[Api.eval]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Eval"],
    responses=standard_responses,
)

router_v1alpha = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
    tags=["Eval"],
    responses=standard_responses,
)


@router.post(
    "/eval/benchmarks/{benchmark_id}/jobs",
    response_model=Job,
    summary="Run an evaluation on a benchmark",
    description="Run an evaluation on a benchmark",
    deprecated=True,
)
@router_v1alpha.post(
    "/eval/benchmarks/{{benchmark_id}}/jobs",
    response_model=Job,
    summary="Run an evaluation on a benchmark",
    description="Run an evaluation on a benchmark",
)
async def run_eval(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to run the evaluation on")],
    benchmark_config: BenchmarkConfig = Body(...),
    svc: EvalService = Depends(get_eval_service),
) -> Job:
    """Run an evaluation on a benchmark."""
    return await svc.run_eval(benchmark_id=benchmark_id, benchmark_config=benchmark_config)


@router.post(
    "/eval/benchmarks/{benchmark_id}/evaluations",
    response_model=EvaluateResponse,
    summary="Evaluate a list of rows on a benchmark",
    description="Evaluate a list of rows on a benchmark",
    deprecated=True,
)
@router_v1alpha.post(
    "/eval/benchmarks/{{benchmark_id}}/evaluations",
    response_model=EvaluateResponse,
    summary="Evaluate a list of rows on a benchmark",
    description="Evaluate a list of rows on a benchmark",
)
async def evaluate_rows(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to run the evaluation on")],
    body: EvaluateRowsRequest = Body(...),
    svc: EvalService = Depends(get_eval_service),
) -> EvaluateResponse:
    """Evaluate a list of rows on a benchmark."""
    return await svc.evaluate_rows(
        benchmark_id=benchmark_id,
        input_rows=body.input_rows,
        scoring_functions=body.scoring_functions,
        benchmark_config=body.benchmark_config,
    )


@router.get(
    "/eval/benchmarks/{benchmark_id}/jobs/{job_id}",
    response_model=Job,
    summary="Get the status of a job",
    description="Get the status of a job",
    deprecated=True,
)
@router_v1alpha.get(
    "/eval/benchmarks/{{benchmark_id}}/jobs/{{job_id}}",
    response_model=Job,
    summary="Get the status of a job",
    description="Get the status of a job",
)
async def job_status(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to run the evaluation on")],
    job_id: Annotated[str, FastAPIPath(..., description="The ID of the job to get the status of")],
    svc: EvalService = Depends(get_eval_service),
) -> Job:
    """Get the status of a job."""
    return await svc.job_status(benchmark_id=benchmark_id, job_id=job_id)


@router.delete(
    "/eval/benchmarks/{benchmark_id}/jobs/{job_id}",
    response_model=None,
    status_code=204,
    summary="Cancel a job",
    description="Cancel a job",
    deprecated=True,
)
@router_v1alpha.delete(
    "/eval/benchmarks/{{benchmark_id}}/jobs/{{job_id}}",
    response_model=None,
    status_code=204,
    summary="Cancel a job",
    description="Cancel a job",
)
async def job_cancel(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to run the evaluation on")],
    job_id: Annotated[str, FastAPIPath(..., description="The ID of the job to cancel")],
    svc: EvalService = Depends(get_eval_service),
) -> None:
    """Cancel a job."""
    await svc.job_cancel(benchmark_id=benchmark_id, job_id=job_id)


@router.get(
    "/eval/benchmarks/{benchmark_id}/jobs/{job_id}/result",
    response_model=EvaluateResponse,
    summary="Get the result of a job",
    description="Get the result of a job",
    deprecated=True,
)
@router_v1alpha.get(
    "/eval/benchmarks/{{benchmark_id}}/jobs/{{job_id}}/result",
    response_model=EvaluateResponse,
    summary="Get the result of a job",
    description="Get the result of a job",
)
async def job_result(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to run the evaluation on")],
    job_id: Annotated[str, FastAPIPath(..., description="The ID of the job to get the result of")],
    svc: EvalService = Depends(get_eval_service),
) -> EvaluateResponse:
    """Get the result of a job."""
    return await svc.job_result(benchmark_id=benchmark_id, job_id=job_id)


# For backward compatibility with the router registry system
def create_eval_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Eval API (legacy compatibility)."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(router_v1alpha)
    return main_router


# Register the router factory
register_router(Api.eval, create_eval_router)
