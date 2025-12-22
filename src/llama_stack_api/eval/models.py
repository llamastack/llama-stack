# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack_api.inference import SamplingParams, SystemMessage
from llama_stack_api.schema_utils import json_schema_type
from llama_stack_api.scoring import ScoringResult
from llama_stack_api.scoring_functions import ScoringFnParams


@json_schema_type
class ModelCandidate(BaseModel):
    """A model candidate for evaluation."""

    type: Literal["model"] = "model"
    model: str = Field(..., description="The model ID to evaluate", min_length=1)
    sampling_params: SamplingParams = Field(..., description="The sampling parameters for the model")
    system_message: SystemMessage | None = Field(
        None, description="The system message providing instructions or context to the model"
    )


EvalCandidate = ModelCandidate


@json_schema_type
class BenchmarkConfig(BaseModel):
    """A benchmark configuration for evaluation."""

    eval_candidate: EvalCandidate = Field(..., description="The candidate to evaluate")
    scoring_params: dict[str, ScoringFnParams] = Field(
        default_factory=dict,
        description="Map between scoring function id and parameters for each scoring function you want to run",
    )
    num_examples: int | None = Field(
        None,
        description="Number of examples to evaluate (useful for testing), if not provided, all examples in the dataset will be evaluated",
        ge=1,
    )
    # we could optinally add any specific dataset config here


@json_schema_type
class EvaluateResponse(BaseModel):
    """The response from an evaluation."""

    generations: list[dict[str, Any]] = Field(..., description="The generations from the evaluation")
    scores: dict[str, ScoringResult] = Field(
        ..., description="The scores from the evaluation. Each key in the dict is a scoring function name"
    )


@json_schema_type
class BenchmarkIdRequest(BaseModel):
    """Request model containing benchmark_id path parameter."""

    benchmark_id: str = Field(..., description="The ID of the benchmark", min_length=1)


@json_schema_type
class RunEvalRequest(BaseModel):
    """Request model for running an evaluation on a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to run the evaluation on", min_length=1)
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark")


@json_schema_type
class RunEvalBodyRequest(BaseModel):
    """Request body model for running an evaluation (without path parameter)."""

    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark")


@json_schema_type
class EvaluateRowsRequest(BaseModel):
    """Request model for evaluating a list of rows on a benchmark."""

    benchmark_id: str = Field(..., description="The ID of the benchmark to run the evaluation on", min_length=1)
    input_rows: list[dict[str, Any]] = Field(..., description="The rows to evaluate", min_length=1)
    scoring_functions: list[str] = Field(
        ..., description="The scoring functions to use for the evaluation", min_length=1
    )
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark")


@json_schema_type
class EvaluateRowsBodyRequest(BaseModel):
    """Request body model for evaluating rows (without path parameter)."""

    input_rows: list[dict[str, Any]] = Field(..., description="The rows to evaluate", min_length=1)
    scoring_functions: list[str] = Field(
        ..., description="The scoring functions to use for the evaluation", min_length=1
    )
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark")


@json_schema_type
class JobStatusRequest(BaseModel):
    """Request model for getting the status of a job."""

    benchmark_id: str = Field(..., description="The ID of the benchmark associated with the job", min_length=1)
    job_id: str = Field(..., description="The ID of the job to get the status of", min_length=1)


@json_schema_type
class JobCancelRequest(BaseModel):
    """Request model for canceling a job."""

    benchmark_id: str = Field(..., description="The ID of the benchmark associated with the job", min_length=1)
    job_id: str = Field(..., description="The ID of the job to cancel", min_length=1)


@json_schema_type
class JobResultRequest(BaseModel):
    """Request model for getting the result of a job."""

    benchmark_id: str = Field(..., description="The ID of the benchmark associated with the job", min_length=1)
    job_id: str = Field(..., description="The ID of the job to get the result of", min_length=1)


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
]
