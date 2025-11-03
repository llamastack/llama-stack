# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import Benchmark, ListBenchmarksResponse


@runtime_checkable
@trace_protocol
class BenchmarksService(Protocol):
    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """List all benchmarks."""
        ...

    async def get_benchmark(
        self,
        benchmark_id: str,
    ) -> Benchmark:
        """Get a benchmark by its ID."""
        ...

    async def register_benchmark(
        self,
        benchmark_id: str,
        dataset_id: str,
        scoring_functions: list[str],
        provider_benchmark_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a benchmark."""
        ...

    async def unregister_benchmark(self, benchmark_id: str) -> None:
        """Unregister a benchmark."""
        ...
