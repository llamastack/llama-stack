# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.apis.version import LLAMA_STACK_API_V1ALPHA
from llama_stack.schema_utils import json_schema_type, webmethod


class CommonBenchmarkFields(BaseModel):
    dataset_id: str
    scoring_functions: list[str]
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this evaluation task",
    )


@json_schema_type
class Benchmark(CommonBenchmarkFields, Resource):
    """A benchmark resource for evaluating model performance.

    :param dataset_id: Identifier of the dataset to use for the benchmark evaluation
    :param scoring_functions: List of scoring function identifiers to apply during evaluation
    :param metadata: Metadata for this evaluation task
    :param type: The resource type, always benchmark
    """

    type: Literal[ResourceType.benchmark] = ResourceType.benchmark

    @property
    def benchmark_id(self) -> str:
        return self.identifier

    @property
    def provider_benchmark_id(self) -> str | None:
        return self.provider_resource_id


class BenchmarkInput(CommonBenchmarkFields, BaseModel):
    benchmark_id: str
    provider_id: str | None = None
    provider_benchmark_id: str | None = None


class ListBenchmarksResponse(BaseModel):
    data: list[Benchmark]


@runtime_checkable
class Benchmarks(Protocol):
    @webmethod(route="/eval/benchmarks", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """List all benchmarks.

        :returns: A ListBenchmarksResponse.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_benchmark(
        self,
        benchmark_id: str,
    ) -> Benchmark:
        """Get a benchmark by its ID.

        :param benchmark_id: The ID of the benchmark to get.
        :returns: A Benchmark.
        """
        ...

    @webmethod(route="/eval/benchmarks", method="POST", level=LLAMA_STACK_API_V1ALPHA, deprecated=True)
    async def register_benchmark(
        self,
        benchmark_id: str,
        dataset_id: str,
        scoring_functions: list[str],
        provider_benchmark_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a benchmark.

        :param benchmark_id: The ID of the benchmark to register.
        :param dataset_id: The ID of the dataset to use for the benchmark.
        :param scoring_functions: The scoring functions to use for the benchmark.
        :param provider_benchmark_id: The ID of the provider benchmark to use for the benchmark.
        :param provider_id: The ID of the provider to use for the benchmark.
        :param metadata: The metadata to use for the benchmark.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}", method="DELETE", level=LLAMA_STACK_API_V1ALPHA, deprecated=True)
    async def unregister_benchmark(self, benchmark_id: str) -> None:
        """Unregister a benchmark.

        :param benchmark_id: The ID of the benchmark to unregister.
        """
        ...
