# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import (
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreRequest,
    ScoreResponse,
    ScoringResult,
    ScoringResultRow,
)
from .scoring_service import ScoringFunctionStore, ScoringService

# Backward compatibility - export Scoring as alias for ScoringService
Scoring = ScoringService

__all__ = [
    "Scoring",
    "ScoringService",
    "ScoringFunctionStore",
    "ScoreBatchRequest",
    "ScoreBatchResponse",
    "ScoreRequest",
    "ScoreResponse",
    "ScoringResult",
    "ScoringResultRow",
]
