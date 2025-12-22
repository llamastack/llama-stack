# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Eval API requests and responses.

This module defines the request and response models for the Eval API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack_api.common.job_types import Job
from llama_stack_api.inference import SamplingParams, SystemMessage
from llama_stack_api.schema_utils import json_schema_type
from llama_stack_api.scoring import ScoringResult
from llama_stack_api.scoring_functions import ScoringFnParams


@json_schema_type
class ModelCandidate(BaseModel):
    """A model candidate for evaluation.

    :param model: The model ID to evaluate.
    :param sampling_params: The sampling parameters for the model.
    :param system_message: (Optional) The system message providing instructions or context to the model.
    """

    type: Literal["model"] = "model"
    model: str
    sampling_params: SamplingParams
    system_message: SystemMessage | None = None


EvalCandidate = ModelCandidate


@json_schema_type
class BenchmarkConfig(BaseModel):
    """A benchmark configuration for evaluation.

    :param eval_candidate: The candidate to evaluate.
    :param scoring_params: Map between scoring function id and parameters for each scoring function you want to run
    :param num_examples: (Optional) The number of examples to evaluate. If not provided, all examples in the dataset will be evaluated
    """

    eval_candidate: EvalCandidate
    scoring_params: dict[str, ScoringFnParams] = Field(
        description="Map between scoring function id and parameters for each scoring function you want to run",
        default_factory=dict,
    )
    num_examples: int | None = Field(
        description="Number of examples to evaluate (useful for testing), if not provided, all examples in the dataset will be evaluated",
        default=None,
    )
    # we could optinally add any specific dataset config here


@json_schema_type
class EvaluateResponse(BaseModel):
    """The response from an evaluation.

    :param generations: The generations from the evaluation.
    :param scores: The scores from the evaluation.
    """

    generations: list[dict[str, Any]]
    # each key in the dict is a scoring function name
    scores: dict[str, ScoringResult]


@json_schema_type
class BenchmarkIdRequest(BaseModel):
    """Request model containing benchmark_id path parameter."""

    benchmark_id: str = Field(..., description="The ID of the benchmark.")


@json_schema_type
class RunEvalRequest(BaseModel):
    """Request model for running an evaluation on a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to run the evaluation on.")
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark.")


@json_schema_type
class RunEvalBodyRequest(BaseModel):
    """Request body model for running an evaluation (without path parameter)."""

    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark.")


@json_schema_type
class EvaluateRowsRequest(BaseModel):
    """Request model for evaluating a list of rows on a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to run the evaluation on.")
    input_rows: list[dict[str, Any]] = Field(..., description="The rows to evaluate.")
    scoring_functions: list[str] = Field(..., description="The scoring functions to use for the evaluation.")
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark.")


@json_schema_type
class EvaluateRowsBodyRequest(BaseModel):
    """Request body model for evaluating rows (without path parameter)."""

    input_rows: list[dict[str, Any]] = Field(..., description="The rows to evaluate.")
    scoring_functions: list[str] = Field(..., description="The scoring functions to use for the evaluation.")
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark.")


@json_schema_type
class JobStatusRequest(BaseModel):
    """Request model for getting the status of a job."""

    benchmark_id: str = Field(..., description="The ID of the benchmark associated with the job.")
    job_id: str = Field(..., description="The ID of the job to get the status of.")


@json_schema_type
class JobCancelRequest(BaseModel):
    """Request model for canceling a job."""

    benchmark_id: str = Field(..., description="The ID of the benchmark associated with the job.")
    job_id: str = Field(..., description="The ID of the job to cancel.")


@json_schema_type
class JobResultRequest(BaseModel):
    """Request model for getting the result of a job."""

    benchmark_id: str = Field(..., description="The ID of the benchmark associated with the job.")
    job_id: str = Field(..., description="The ID of the job to get the result of.")


__all__ = [
    "ModelCandidate",
    "EvalCandidate",
    "BenchmarkConfig",
    "EvaluateResponse",
    "BenchmarkIdRequest",
    "RunEvalRequest",
    "RunEvalBodyRequest",
    "EvaluateRowsRequest",
    "EvaluateRowsBodyRequest",
    "JobStatusRequest",
    "JobCancelRequest",
    "JobResultRequest",
    "Job",
]
