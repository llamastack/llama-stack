# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Eval API protocol and models.

This module contains the Eval protocol definition.
Pydantic models are defined in llama_stack_api.eval.models.
The FastAPI router is defined in llama_stack_api.eval.fastapi_routes.
"""

from llama_stack_api.common.job_types import Job

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Eval

# Import models for re-export
from .models import (
    BenchmarkConfig,
    BenchmarkIdRequest,
    EvalCandidate,
    EvaluateResponse,
    EvaluateRowsBodyRequest,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    ModelCandidate,
    RunEvalBodyRequest,
    RunEvalRequest,
)

__all__ = [
    "Eval",
    "BenchmarkConfig",
    "BenchmarkIdRequest",
    "EvalCandidate",
    "EvaluateResponse",
    "EvaluateRowsBodyRequest",
    "EvaluateRowsRequest",
    "Job",
    "JobCancelRequest",
    "JobResultRequest",
    "JobStatusRequest",
    "ModelCandidate",
    "RunEvalBodyRequest",
    "RunEvalRequest",
    "fastapi_routes",
]
