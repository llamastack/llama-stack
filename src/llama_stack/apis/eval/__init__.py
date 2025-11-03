# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .eval_service import EvalService
from .models import (
    AgentCandidate,
    BenchmarkConfig,
    EvalCandidate,
    EvaluateResponse,
    EvaluateRowsRequest,
    ModelCandidate,
)

# Backward compatibility - export Eval as alias for EvalService
Eval = EvalService

__all__ = [
    "Eval",
    "EvalService",
    "ModelCandidate",
    "AgentCandidate",
    "EvalCandidate",
    "BenchmarkConfig",
    "EvaluateResponse",
    "EvaluateRowsRequest",
]
