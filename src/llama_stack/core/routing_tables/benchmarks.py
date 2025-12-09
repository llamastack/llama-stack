# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.core.datatypes import (
    BenchmarkWithOwner,
)
from llama_stack.log import get_logger
from llama_stack_api import (
    Benchmark,
    Benchmarks,
    GetBenchmarkRequest,
    ListBenchmarksRequest,
    ListBenchmarksResponse,
    RegisterBenchmarkRequest,
    UnregisterBenchmarkRequest,
)

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class BenchmarksRoutingTable(CommonRoutingTableImpl, Benchmarks):
    async def list_benchmarks(self, request: ListBenchmarksRequest) -> ListBenchmarksResponse:
        return ListBenchmarksResponse(data=await self.get_all_with_type("benchmark"))

    async def get_benchmark(self, request: GetBenchmarkRequest) -> Benchmark:
        benchmark = await self.get_object_by_identifier("benchmark", request.benchmark_id)
        if benchmark is None:
            raise ValueError(f"Benchmark '{request.benchmark_id}' not found")
        return benchmark

    async def register_benchmark(
        self,
        request: RegisterBenchmarkRequest,
    ) -> None:
        metadata = request.metadata if request.metadata is not None else {}
        provider_id = request.provider_id
        if provider_id is None:
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        provider_benchmark_id = request.provider_benchmark_id
        if provider_benchmark_id is None:
            provider_benchmark_id = request.benchmark_id
        benchmark = BenchmarkWithOwner(
            identifier=request.benchmark_id,
            dataset_id=request.dataset_id,
            scoring_functions=request.scoring_functions,
            metadata=metadata,
            provider_id=provider_id,
            provider_resource_id=provider_benchmark_id,
        )
        await self.register_object(benchmark)

    async def unregister_benchmark(self, request: UnregisterBenchmarkRequest) -> None:
        get_request = GetBenchmarkRequest(benchmark_id=request.benchmark_id)
        existing_benchmark = await self.get_benchmark(get_request)
        await self.unregister_object(existing_benchmark)
