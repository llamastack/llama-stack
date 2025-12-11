# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Benchmarks API requests and responses.

This module defines the request and response models for the Benchmarks API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class ListBenchmarksRequest(BaseModel):
    """Request model for listing benchmarks."""

    pass


@json_schema_type
class GetBenchmarkRequest(BaseModel):
    """Request model for getting a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to get.")


@json_schema_type
class RegisterBenchmarkRequest(BaseModel):
    """Request model for registering a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to register.")
    dataset_id: str = Field(..., description="The ID of the dataset to use for the benchmark.")
    scoring_functions: list[str] = Field(..., description="The scoring functions to use for the benchmark.")
    provider_benchmark_id: str | None = Field(
        default=None, description="The ID of the provider benchmark to use for the benchmark."
    )
    provider_id: str | None = Field(default=None, description="The ID of the provider to use for the benchmark.")
    metadata: dict[str, Any] | None = Field(default=None, description="The metadata to use for the benchmark.")


@json_schema_type
class UnregisterBenchmarkRequest(BaseModel):
    """Request model for unregistering a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to unregister.")


class CommonBenchmarkFields(BaseModel):
    dataset_id: str = Field(..., description="Identifier of the dataset to use for the benchmark evaluation.")
    scoring_functions: list[str] = Field(
        ..., description="List of scoring function identifiers to apply during evaluation."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for this evaluation task.",
    )


@json_schema_type
class Benchmark(CommonBenchmarkFields, Resource):
    """A benchmark resource for evaluating model performance."""

    type: Literal[ResourceType.benchmark] = Field(
        default=ResourceType.benchmark,
        description="The resource type, always benchmark.",
    )

    @property
    def benchmark_id(self) -> str:
        return self.identifier

    @property
    def provider_benchmark_id(self) -> str | None:
        return self.provider_resource_id


class BenchmarkInput(CommonBenchmarkFields, BaseModel):
    benchmark_id: str = Field(..., description="The ID of the benchmark.")
    provider_id: str | None = Field(default=None, description="The ID of the provider to use for the benchmark.")
    provider_benchmark_id: str | None = Field(
        default=None, description="The ID of the provider benchmark to use for the benchmark."
    )


@json_schema_type
class ListBenchmarksResponse(BaseModel):
    """Response containing a list of benchmark objects."""

    data: list[Benchmark] = Field(..., description="List of benchmark objects.")


__all__ = [
    "ListBenchmarksRequest",
    "GetBenchmarkRequest",
    "RegisterBenchmarkRequest",
    "UnregisterBenchmarkRequest",
    "CommonBenchmarkFields",
    "Benchmark",
    "BenchmarkInput",
    "ListBenchmarksResponse",
]
