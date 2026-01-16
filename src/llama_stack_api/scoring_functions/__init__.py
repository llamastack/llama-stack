# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""ScoringFunctions API protocol and models.

This module contains the ScoringFunctions protocol definition.
Pydantic models are defined in llama_stack_api.scoring_functions.models.
The FastAPI router is defined in llama_stack_api.scoring_functions.fastapi_routes.
"""

from . import fastapi_routes
from .api import ScoringFunctions
from .models import (
    AggregationFunctionType,
    BasicScoringFnParams,
    CommonScoringFnFields,
    GetScoringFunctionRequest,
    ListScoringFunctionsRequest,
    ListScoringFunctionsResponse,
    LLMAsJudgeScoringFnParams,
    RegexParserScoringFnParams,
    RegisterScoringFunctionRequest,
    ScoringFn,
    ScoringFnInput,
    ScoringFnParams,
    ScoringFnParamsType,
    UnregisterScoringFunctionRequest,
)

__all__ = [
    "ScoringFunctions",
    "ScoringFn",
    "ScoringFnInput",
    "ScoringFnParams",
    "ScoringFnParamsType",
    "AggregationFunctionType",
    "LLMAsJudgeScoringFnParams",
    "RegexParserScoringFnParams",
    "BasicScoringFnParams",
    "CommonScoringFnFields",
    "ListScoringFunctionsResponse",
    "ListScoringFunctionsRequest",
    "GetScoringFunctionRequest",
    "RegisterScoringFunctionRequest",
    "UnregisterScoringFunctionRequest",
    "fastapi_routes",
]
