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

from typing import Any, Protocol, runtime_checkable

from llama_stack_api.scoring_functions import ScoringFn, ScoringFnParams

from .models import ScoreBatchResponse, ScoreResponse


class ScoringFunctionStore(Protocol):
    """Protocol for storing and retrieving scoring functions."""

    def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn: ...


@runtime_checkable
class Scoring(Protocol):
    """Protocol for scoring operations."""

    scoring_function_store: ScoringFunctionStore

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: dict[str, ScoringFnParams | None],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        """Score a batch of rows.

        :param dataset_id: The ID of the dataset to score.
        :param scoring_functions: The scoring functions to use for the scoring.
        :param save_results_dataset: Whether to save the results to a dataset.
        :returns: A ScoreBatchResponse.
        """
        ...

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_functions: dict[str, ScoringFnParams | None],
    ) -> ScoreResponse:
        """Score a list of rows.

        :param input_rows: The rows to score.
        :param scoring_functions: The scoring functions to use for the scoring.
        :returns: A ScoreResponse object containing rows and aggregated results.
        """
        ...
