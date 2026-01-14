# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Scoring API protocol definition.

This module contains the Scoring protocol definition.
Pydantic models are defined in llama_stack_api.scoring.models.
The FastAPI router is defined in llama_stack_api.scoring.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from llama_stack_api.scoring_functions import ScoringFn

from .models import ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse


class ScoringFunctionStore(Protocol):
    """Protocol for storing and retrieving scoring functions."""

    def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn: ...


@runtime_checkable
class Scoring(Protocol):
    """Protocol for scoring operations."""

    scoring_function_store: ScoringFunctionStore

    async def score_batch(self, request: ScoreBatchRequest) -> ScoreBatchResponse: ...

    async def score(self, request: ScoreRequest) -> ScoreResponse: ...
