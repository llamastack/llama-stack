# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type


class CommonBenchmarkFields(BaseModel):
    dataset_id: str = Field(..., description="The ID of the dataset to use for the benchmark")
    scoring_functions: list[str] = Field(..., description="The scoring functions to use for the benchmark")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this evaluation task",
    )


@json_schema_type
class Benchmark(CommonBenchmarkFields, Resource):
    """A benchmark resource for evaluating model performance."""

    type: Literal[ResourceType.benchmark] = Field(
        default=ResourceType.benchmark, description="The resource type, always benchmark"
    )


class ListBenchmarksResponse(BaseModel):
    """Response model for listing benchmarks."""

    data: list[Benchmark] = Field(..., description="List of benchmark resources")


@json_schema_type
class RegisterBenchmarkRequest(BaseModel):
    """Request model for registering a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to register")
    dataset_id: str = Field(..., description="The ID of the dataset to use for the benchmark")
    scoring_functions: list[str] = Field(..., description="The scoring functions to use for the benchmark")
    provider_benchmark_id: str | None = Field(
        default=None, description="The ID of the provider benchmark to use for the benchmark"
    )
    provider_id: str | None = Field(default=None, description="The ID of the provider to use for the benchmark")
    metadata: dict[str, Any] | None = Field(default=None, description="The metadata to use for the benchmark")


class BenchmarkInput(CommonBenchmarkFields, BaseModel):
    benchmark_id: str = Field(..., description="The ID of the benchmark to use for the benchmark")
    provider_id: str | None = Field(default=None, description="The ID of the provider to use for the benchmark")
    provider_benchmark_id: str | None = Field(
        default=None, description="The ID of the provider benchmark to use for the benchmark"
    )
