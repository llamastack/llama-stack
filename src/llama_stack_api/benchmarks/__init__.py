# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Benchmarks API protocol and models.

This module contains the Benchmarks protocol definition.
Pydantic models are defined in llama_stack_api.benchmarks.models.
The FastAPI router is defined in llama_stack_api.benchmarks.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Benchmarks

# Import models for re-export
from .models import (
    Benchmark,
    BenchmarkInput,
    CommonBenchmarkFields,
    GetBenchmarkRequest,
    ListBenchmarksRequest,
    ListBenchmarksResponse,
    RegisterBenchmarkRequest,
    UnregisterBenchmarkRequest,
)

__all__ = [
    "Benchmarks",
    "Benchmark",
    "BenchmarkInput",
    "CommonBenchmarkFields",
    "ListBenchmarksResponse",
    "ListBenchmarksRequest",
    "GetBenchmarkRequest",
    "RegisterBenchmarkRequest",
    "UnregisterBenchmarkRequest",
    "fastapi_routes",
]
