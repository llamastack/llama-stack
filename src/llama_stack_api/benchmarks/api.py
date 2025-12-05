# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    Benchmark,
    GetBenchmarkRequest,
    ListBenchmarksRequest,
    ListBenchmarksResponse,
    RegisterBenchmarkRequest,
    UnregisterBenchmarkRequest,
)


@runtime_checkable
class Benchmarks(Protocol):
    async def list_benchmarks(
        self,
        request: ListBenchmarksRequest,
    ) -> ListBenchmarksResponse: ...

    async def get_benchmark(
        self,
        request: GetBenchmarkRequest,
    ) -> Benchmark: ...

    async def register_benchmark(
        self,
        request: RegisterBenchmarkRequest,
    ) -> None: ...

    async def unregister_benchmark(
        self,
        request: UnregisterBenchmarkRequest,
    ) -> None: ...
