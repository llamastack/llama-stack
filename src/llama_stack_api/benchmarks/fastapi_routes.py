# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Benchmarks API.

This module defines the FastAPI router for the Benchmarks API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from llama_stack_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA

from .api import Benchmarks
from .models import (
    Benchmark,
    GetBenchmarkRequest,
    ListBenchmarksRequest,
    ListBenchmarksResponse,
    RegisterBenchmarkRequest,
    UnregisterBenchmarkRequest,
)

# Automatically generate dependency functions from Pydantic models
# This ensures the models are the single source of truth for descriptions
get_list_benchmarks_request = create_query_dependency(ListBenchmarksRequest)
get_get_benchmark_request = create_path_dependency(GetBenchmarkRequest)
get_unregister_benchmark_request = create_path_dependency(UnregisterBenchmarkRequest)


def create_router(impl: Benchmarks) -> APIRouter:
    """Create a FastAPI router for the Benchmarks API.

    Args:
        impl: The Benchmarks implementation instance

    Returns:
        APIRouter configured for the Benchmarks API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
        tags=["Benchmarks"],
        responses=standard_responses,
    )

    @router.get(
        "/eval/benchmarks",
        response_model=ListBenchmarksResponse,
        summary="List all benchmarks.",
        description="List all benchmarks.",
        responses={
            200: {"description": "A ListBenchmarksResponse."},
        },
    )
    async def list_benchmarks(
        request: Annotated[ListBenchmarksRequest, Depends(get_list_benchmarks_request)],
    ) -> ListBenchmarksResponse:
        return await impl.list_benchmarks(request)

    @router.get(
        "/eval/benchmarks/{benchmark_id}",
        response_model=Benchmark,
        summary="Get a benchmark by its ID.",
        description="Get a benchmark by its ID.",
        responses={
            200: {"description": "A Benchmark."},
        },
    )
    async def get_benchmark(
        request: Annotated[GetBenchmarkRequest, Depends(get_get_benchmark_request)],
    ) -> Benchmark:
        return await impl.get_benchmark(request)

    @router.post(
        "/eval/benchmarks",
        summary="Register a benchmark.",
        description="Register a benchmark.",
        responses={
            200: {"description": "The benchmark was successfully registered."},
        },
        deprecated=True,
    )
    async def register_benchmark(
        request: Annotated[RegisterBenchmarkRequest, Body(...)],
    ) -> None:
        return await impl.register_benchmark(request)

    @router.delete(
        "/eval/benchmarks/{benchmark_id}",
        summary="Unregister a benchmark.",
        description="Unregister a benchmark.",
        responses={
            200: {"description": "The benchmark was successfully unregistered."},
        },
        deprecated=True,
    )
    async def unregister_benchmark(
        request: Annotated[UnregisterBenchmarkRequest, Depends(get_unregister_benchmark_request)],
    ) -> None:
        return await impl.unregister_benchmark(request)

    return router
