# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.benchmarks import Benchmark, Benchmarks, ListBenchmarksResponse
from llama_stack.core.datatypes import (
    BenchmarkWithOwner,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class BenchmarksRoutingTable(CommonRoutingTableImpl, Benchmarks):
    async def list_benchmarks(self) -> ListBenchmarksResponse:
        return ListBenchmarksResponse(data=await self.get_all_with_type("benchmark"))

    async def get_benchmark(self, benchmark_id: str) -> Benchmark:
        benchmark = await self.get_object_by_identifier("benchmark", benchmark_id)
        if benchmark is None:
            raise ValueError(f"Benchmark '{benchmark_id}' not found")
        return benchmark

    async def register_benchmark(
        self,
        benchmark_id: str,
        dataset_id: str,
        scoring_functions: list[str],
        metadata: dict[str, Any] | None = None,
        provider_benchmark_id: str | None = None,
        provider_id: str | None = None,
    ) -> None:
        if metadata is None:
            metadata = {}
        if provider_id is None:
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        if provider_benchmark_id is None:
            provider_benchmark_id = benchmark_id
        benchmark = BenchmarkWithOwner(
            identifier=benchmark_id,
            dataset_id=dataset_id,
            scoring_functions=scoring_functions,
            metadata=metadata,
            provider_id=provider_id,
            provider_resource_id=provider_benchmark_id,
        )
        await self.register_object(benchmark)

    async def unregister_benchmark(self, benchmark_id: str) -> None:
        existing_benchmark = await self.get_benchmark(benchmark_id)
        await self.unregister_object(existing_benchmark)
