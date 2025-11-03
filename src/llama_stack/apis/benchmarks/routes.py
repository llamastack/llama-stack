# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .benchmarks_service import BenchmarksService
from .models import Benchmark, ListBenchmarksResponse, RegisterBenchmarkRequest


def get_benchmarks_service(request: Request) -> BenchmarksService:
    """Dependency to get the benchmarks service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.benchmarks not in impls:
        raise ValueError("Benchmarks API implementation not found")
    return impls[Api.benchmarks]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Benchmarks"],
    responses=standard_responses,
)

router_v1alpha = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
    tags=["Benchmarks"],
    responses=standard_responses,
)


@router.get(
    "/eval/benchmarks",
    response_model=ListBenchmarksResponse,
    summary="List all benchmarks",
    description="List all benchmarks",
    deprecated=True,
)
@router_v1alpha.get(
    "/eval/benchmarks",
    response_model=ListBenchmarksResponse,
    summary="List all benchmarks",
    description="List all benchmarks",
)
async def list_benchmarks(svc: BenchmarksService = Depends(get_benchmarks_service)) -> ListBenchmarksResponse:
    """List all benchmarks."""
    return await svc.list_benchmarks()


@router.get(
    "/eval/benchmarks/{benchmark_id}",
    response_model=Benchmark,
    summary="Get a benchmark by its ID",
    description="Get a benchmark by its ID",
    deprecated=True,
)
@router_v1alpha.get(
    "/eval/benchmarks/{{benchmark_id}}",
    response_model=Benchmark,
    summary="Get a benchmark by its ID",
    description="Get a benchmark by its ID",
)
async def get_benchmark(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to get")],
    svc: BenchmarksService = Depends(get_benchmarks_service),
) -> Benchmark:
    """Get a benchmark by its ID."""
    return await svc.get_benchmark(benchmark_id=benchmark_id)


@router.post(
    "/eval/benchmarks",
    response_model=None,
    status_code=204,
    summary="Register a benchmark",
    description="Register a benchmark",
    deprecated=True,
)
@router_v1alpha.post(
    "/eval/benchmarks",
    response_model=None,
    status_code=204,
    summary="Register a benchmark",
    description="Register a benchmark",
)
async def register_benchmark(
    body: RegisterBenchmarkRequest = Body(...),
    svc: BenchmarksService = Depends(get_benchmarks_service),
) -> None:
    """Register a benchmark."""
    return await svc.register_benchmark(
        benchmark_id=body.benchmark_id,
        dataset_id=body.dataset_id,
        scoring_functions=body.scoring_functions,
        provider_benchmark_id=body.provider_benchmark_id,
        provider_id=body.provider_id,
        metadata=body.metadata,
    )


@router.delete(
    "/eval/benchmarks/{benchmark_id}",
    response_model=None,
    status_code=204,
    summary="Unregister a benchmark",
    description="Unregister a benchmark",
    deprecated=True,
)
@router_v1alpha.delete(
    "/eval/benchmarks/{{benchmark_id}}",
    response_model=None,
    status_code=204,
    summary="Unregister a benchmark",
    description="Unregister a benchmark",
)
async def unregister_benchmark(
    benchmark_id: Annotated[str, FastAPIPath(..., description="The ID of the benchmark to unregister")],
    svc: BenchmarksService = Depends(get_benchmarks_service),
) -> None:
    """Unregister a benchmark."""
    await svc.unregister_benchmark(benchmark_id=benchmark_id)


# For backward compatibility with the router registry system
def create_benchmarks_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Benchmarks API (legacy compatibility)."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(router_v1alpha)
    return main_router


# Register the router factory
register_router(Api.benchmarks, create_benchmarks_router)
