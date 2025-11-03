# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.apis.common.job_types import Job
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import BenchmarkConfig, EvaluateResponse


@runtime_checkable
@trace_protocol
class EvalService(Protocol):
    """Evaluations

    Llama Stack Evaluation API for running evaluations on model and agent candidates."""

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        """Run an evaluation on a benchmark."""
        ...

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark."""
        ...

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        """Get the status of a job."""
        ...

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job."""
        ...

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job."""
        ...
