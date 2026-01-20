# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from llama_stack_api.common.job_types import Job

from .models import (
    EvaluateResponse,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    RunEvalRequest,
)


@runtime_checkable
class Eval(Protocol):
    """Evaluations

    Llama Stack Evaluation API for running evaluations on model and agent candidates."""

    async def run_eval(
        self,
        request: RunEvalRequest,
    ) -> Job:
        """Run an evaluation on a benchmark."""
        ...

    async def evaluate_rows(
        self,
        request: EvaluateRowsRequest,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark."""
        ...

    async def job_status(self, request: JobStatusRequest) -> Job:
        """Get the status of a job."""
        ...

    async def job_cancel(self, request: JobCancelRequest) -> None:
        """Cancel a job."""
        ...

    async def job_result(self, request: JobResultRequest) -> EvaluateResponse:
        """Get the result of a job."""
        ...
