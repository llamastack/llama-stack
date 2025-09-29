# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnParams
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, webmethod

# mapping of metric to value
ScoringResultRow = dict[str, Any]


@json_schema_type
class ScoringResult(BaseModel):
    """
    A scoring result for a single row.

    :param score_rows: The scoring result for each row. Each row is a map of column name to value.
    :param aggregated_results: Map of metric name to aggregated value
    """

    score_rows: list[ScoringResultRow]
    # aggregated metrics to value
    aggregated_results: dict[str, Any]


@json_schema_type
class ScoreBatchResponse(BaseModel):
    """Response from batch scoring operations on datasets.

    :param dataset_id: (Optional) The identifier of the dataset that was scored
    :param results: A map of scoring function name to ScoringResult
    """

    dataset_id: str | None = None
    results: dict[str, ScoringResult]


@json_schema_type
class ScoreResponse(BaseModel):
    """
    The response from scoring.

    :param results: A map of scoring function name to ScoringResult.
    """

    # each key in the dict is a scoring function name
    results: dict[str, ScoringResult]


class ScoringFunctionStore(Protocol):
    def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn: ...


@runtime_checkable
class Scoring(Protocol):
    scoring_function_store: ScoringFunctionStore

    @webmethod(route="/scoring/score-batch", method="POST", level=LLAMA_STACK_API_V1)
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

    @webmethod(route="/scoring/score", method="POST", level=LLAMA_STACK_API_V1)
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
