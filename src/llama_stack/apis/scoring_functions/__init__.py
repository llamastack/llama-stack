# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import (
    AggregationFunctionType,
    BasicScoringFnParams,
    CommonScoringFnFields,
    ListScoringFunctionsResponse,
    LLMAsJudgeScoringFnParams,
    RegexParserScoringFnParams,
    RegisterScoringFunctionRequest,
    ScoringFn,
    ScoringFnInput,
    ScoringFnParams,
    ScoringFnParamsType,
)
from .scoring_functions_service import ScoringFunctionsService

# Backward compatibility - export ScoringFunctions as alias for ScoringFunctionsService
ScoringFunctions = ScoringFunctionsService

__all__ = [
    "ScoringFunctions",
    "ScoringFunctionsService",
    "ScoringFn",
    "ScoringFnInput",
    "CommonScoringFnFields",
    "ScoringFnParams",
    "ScoringFnParamsType",
    "LLMAsJudgeScoringFnParams",
    "RegexParserScoringFnParams",
    "BasicScoringFnParams",
    "AggregationFunctionType",
    "ListScoringFunctionsResponse",
    "RegisterScoringFunctionRequest",
]
