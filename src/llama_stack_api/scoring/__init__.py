# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Scoring API protocol and models.

This module contains the Scoring protocol definition.
Pydantic models are defined in llama_stack_api.scoring.models.
The FastAPI router is defined in llama_stack_api.scoring.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for FastAPI router
from .api import Scoring, ScoringFunctionStore

# Import models for re-export
from .models import (
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreRequest,
    ScoreResponse,
    ScoringResult,
    ScoringResultRow,
)

# Import scoring_functions for re-export
from .scoring_functions import (
    AggregationFunctionType,
    BasicScoringFnParams,
    CommonScoringFnFields,
    LLMAsJudgeScoringFnParams,
    ListScoringFunctionsResponse,
    RegexParserScoringFnParams,
    ScoringFn,
    ScoringFnInput,
    ScoringFnParams,
    ScoringFnParamsType,
    ScoringFunctions,
)

__all__ = [
    "Scoring",
    "ScoringFunctionStore",
    "ScoringResult",
    "ScoringResultRow",
    "ScoreBatchResponse",
    "ScoreResponse",
    "ScoreRequest",
    "ScoreBatchRequest",
    "AggregationFunctionType",
    "BasicScoringFnParams",
    "CommonScoringFnFields",
    "LLMAsJudgeScoringFnParams",
    "ListScoringFunctionsResponse",
    "RegexParserScoringFnParams",
    "ScoringFn",
    "ScoringFnInput",
    "ScoringFnParams",
    "ScoringFnParamsType",
    "ScoringFunctions",
    "fastapi_routes",
]
