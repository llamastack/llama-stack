# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class ScoringFnParamsType(StrEnum):
    """Types of scoring function parameter configurations."""

    llm_as_judge = "llm_as_judge"
    regex_parser = "regex_parser"
    basic = "basic"


@json_schema_type
class AggregationFunctionType(StrEnum):
    """Types of aggregation functions for scoring results."""

    average = "average"
    weighted_average = "weighted_average"
    median = "median"
    categorical_count = "categorical_count"
    accuracy = "accuracy"


@json_schema_type
class LLMAsJudgeScoringFnParams(BaseModel):
    """Parameters for LLM-as-judge scoring function configuration."""

    type: Literal[ScoringFnParamsType.llm_as_judge] = ScoringFnParamsType.llm_as_judge
    judge_model: str
    prompt_template: str | None = None
    judge_score_regexes: list[str] = Field(
        description="Regexes to extract the answer from generated response",
        default_factory=lambda: [],
    )
    aggregation_functions: list[AggregationFunctionType] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=lambda: [],
    )


@json_schema_type
class RegexParserScoringFnParams(BaseModel):
    """Parameters for regex parser scoring function configuration."""

    type: Literal[ScoringFnParamsType.regex_parser] = ScoringFnParamsType.regex_parser
    parsing_regexes: list[str] = Field(
        description="Regex to extract the answer from generated response",
        default_factory=lambda: [],
    )
    aggregation_functions: list[AggregationFunctionType] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=lambda: [],
    )


@json_schema_type
class BasicScoringFnParams(BaseModel):
    """Parameters for basic scoring function configuration."""

    type: Literal[ScoringFnParamsType.basic] = ScoringFnParamsType.basic
    aggregation_functions: list[AggregationFunctionType] = Field(
        description="Aggregation functions to apply to the scores of each row",
        default_factory=list,
    )


ScoringFnParams = Annotated[
    LLMAsJudgeScoringFnParams | RegexParserScoringFnParams | BasicScoringFnParams,
    Field(discriminator="type"),
]
register_schema(ScoringFnParams, name="ScoringFnParams")


class CommonScoringFnFields(BaseModel):
    description: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this definition",
    )
    return_type: ParamType = Field(
        description="The return type of the deterministic function",
    )
    params: ScoringFnParams | None = Field(
        description="The parameters for the scoring function for benchmark eval, these can be overridden for app eval",
        default=None,
    )


@json_schema_type
class ScoringFn(CommonScoringFnFields, Resource):
    """A scoring function resource for evaluating model outputs."""

    type: Literal[ResourceType.scoring_function] = ResourceType.scoring_function

    @property
    def scoring_fn_id(self) -> str:
        return self.identifier

    @property
    def provider_scoring_fn_id(self) -> str | None:
        return self.provider_resource_id


class ScoringFnInput(CommonScoringFnFields, BaseModel):
    scoring_fn_id: str
    provider_id: str | None = None
    provider_scoring_fn_id: str | None = None


class ListScoringFunctionsResponse(BaseModel):
    """Response model for listing scoring functions."""

    data: list[ScoringFn] = Field(..., description="List of scoring function resources")


@json_schema_type
class RegisterScoringFunctionRequest(BaseModel):
    """Request model for registering a scoring function."""

    scoring_fn_id: str = Field(..., description="The ID of the scoring function to register")
    description: str = Field(..., description="The description of the scoring function")
    return_type: ParamType = Field(..., description="The return type of the scoring function")
    provider_scoring_fn_id: str | None = Field(
        default=None, description="The ID of the provider scoring function to use for the scoring function"
    )
    provider_id: str | None = Field(default=None, description="The ID of the provider to use for the scoring function")
    params: ScoringFnParams | None = Field(
        default=None,
        description="The parameters for the scoring function for benchmark eval, these can be overridden for app eval",
    )
