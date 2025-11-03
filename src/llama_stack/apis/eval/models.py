# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.apis.scoring.models import ScoringResult
from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class ModelCandidate(BaseModel):
    """A model candidate for evaluation."""

    type: Literal["model"] = Field(default="model", description="The type of candidate.")
    model: str = Field(..., description="The model ID to evaluate.")
    sampling_params: SamplingParams = Field(..., description="The sampling parameters for the model.")
    system_message: SystemMessage | None = Field(
        default=None, description="The system message providing instructions or context to the model."
    )


@json_schema_type
class AgentCandidate(BaseModel):
    """An agent candidate for evaluation."""

    type: Literal["agent"] = Field(default="agent", description="The type of candidate.")
    config: AgentConfig = Field(..., description="The configuration for the agent candidate.")


EvalCandidate = Annotated[ModelCandidate | AgentCandidate, Field(discriminator="type")]
register_schema(EvalCandidate, name="EvalCandidate")


@json_schema_type
class BenchmarkConfig(BaseModel):
    """A benchmark configuration for evaluation."""

    eval_candidate: EvalCandidate = Field(..., description="The candidate to evaluate.")
    scoring_params: dict[str, ScoringFnParams] = Field(
        description="Map between scoring function id and parameters for each scoring function you want to run.",
        default_factory=dict,
    )
    num_examples: int | None = Field(
        description="The number of examples to evaluate. If not provided, all examples in the dataset will be evaluated.",
        default=None,
    )


@json_schema_type
class EvaluateResponse(BaseModel):
    """The response from an evaluation."""

    generations: list[dict[str, Any]] = Field(..., description="The generations from the evaluation.")
    scores: dict[str, ScoringResult] = Field(
        ..., description="The scores from the evaluation. Each key in the dict is a scoring function name."
    )


@json_schema_type
class EvaluateRowsRequest(BaseModel):
    """Request model for evaluating rows."""

    input_rows: list[dict[str, Any]] = Field(..., description="The rows to evaluate.")
    scoring_functions: list[str] = Field(..., description="The scoring functions to use for the evaluation.")
    benchmark_config: BenchmarkConfig = Field(..., description="The configuration for the benchmark.")
