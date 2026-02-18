# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Scoring API requests and responses.

This module defines the request and response models for the Scoring API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api.schema_utils import json_schema_type
from llama_stack_api.scoring_functions import ScoringFnParams

# mapping of metric to value
ScoringResultRow = dict[str, Any]


@json_schema_type
class ScoringResult(BaseModel):
    """
    A scoring result for a single row.
    """

    score_rows: list[ScoringResultRow] = Field(
        ..., description="The scoring result for each row. Each row is a map of column name to value."
    )
    aggregated_results: dict[str, Any] = Field(..., description="Map of metric name to aggregated value")


@json_schema_type
class ScoreBatchResponse(BaseModel):
    """Response from batch scoring operations on datasets."""

    dataset_id: str | None = Field(default=None, description="(Optional) The identifier of the dataset that was scored")
    results: dict[str, ScoringResult] = Field(..., description="A map of scoring function name to ScoringResult")


@json_schema_type
class ScoreResponse(BaseModel):
    """
    The response from scoring.
    """

    results: dict[str, ScoringResult] = Field(..., description="A map of scoring function name to ScoringResult.")


@json_schema_type
class ScoreRequest(BaseModel):
    """Request model for scoring a list of rows."""

    input_rows: list[dict[str, Any]] = Field(..., description="The rows to score.")
    scoring_functions: dict[str, ScoringFnParams | None] = Field(
        ..., description="The scoring functions to use for the scoring."
    )


@json_schema_type
class ScoreBatchRequest(BaseModel):
    """Request model for scoring a batch of rows from a dataset."""

    dataset_id: str = Field(..., description="The ID of the dataset to score.")
    scoring_functions: dict[str, ScoringFnParams | None] = Field(
        ..., description="The scoring functions to use for the scoring."
    )
    save_results_dataset: bool = Field(default=False, description="Whether to save the results to a dataset.")


__all__ = [
    "ScoringResult",
    "ScoringResultRow",
    "ScoreBatchResponse",
    "ScoreResponse",
    "ScoreRequest",
    "ScoreBatchRequest",
]
