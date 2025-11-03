# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .benchmarks_service import BenchmarksService
from .models import (
    Benchmark,
    BenchmarkInput,
    CommonBenchmarkFields,
    ListBenchmarksResponse,
    RegisterBenchmarkRequest,
)

# Backward compatibility - export Benchmarks as alias for BenchmarksService
Benchmarks = BenchmarksService

__all__ = [
    "Benchmarks",
    "BenchmarksService",
    "Benchmark",
    "BenchmarkInput",
    "CommonBenchmarkFields",
    "ListBenchmarksResponse",
    "RegisterBenchmarkRequest",
]
