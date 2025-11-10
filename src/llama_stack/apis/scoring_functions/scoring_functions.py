# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# TODO: use enum.StrEnum when we drop support for python 3.10
from enum import StrEnum
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


# Perhaps more structure can be imposed on these functions. Maybe they could be associated
# with standard metrics so they can be rolled up?
@json_schema_type
class ScoringFnParamsType(StrEnum):
    """Types of scoring function parameter configurations.
    :cvar llm_as_judge: Use an LLM model to evaluate and score responses
    :cvar regex_parser: Use regex patterns to extract and score specific parts of responses
    :cvar basic: Basic scoring with simple aggregation functions
    """

    llm_as_judge = "llm_as_judge"
    regex_parser = "regex_parser"
    basic = "basic"


@json_schema_type
class AggregationFunctionType(StrEnum):
    """Types of aggregation functions for scoring results.
    :cvar average: Calculate the arithmetic mean of scores
    :cvar weighted_average: Calculate a weighted average of scores
    :cvar median: Calculate the median value of scores
    :cvar categorical_count: Count occurrences of categorical values
    :cvar accuracy: Calculate accuracy as the proportion of correct answers
    """

    average = "average"
    weighted_average = "weighted_average"
    median = "median"
    categorical_count = "categorical_count"
    accuracy = "accuracy"


@json_schema_type
class LLMAsJudgeScoringFnParams(BaseModel):
    """Parameters for LLM-as-judge scoring function configuration.
    :param type: The type of scoring function parameters, always llm_as_judge
    :param judge_model: Identifier of the LLM model to use as a judge for scoring
    :param prompt_template: (Optional) Custom prompt template for the judge model
    :param judge_score_regexes: Regexes to extract the answer from generated response
    :param aggregation_functions: Aggregation functions to apply to the scores of each row
    """

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
    """Parameters for regex parser scoring function configuration.
    :param type: The type of scoring function parameters, always regex_parser
    :param parsing_regexes: Regex to extract the answer from generated response
    :param aggregation_functions: Aggregation functions to apply to the scores of each row
    """

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
    """Parameters for basic scoring function configuration.
    :param type: The type of scoring function parameters, always basic
    :param aggregation_functions: Aggregation functions to apply to the scores of each row
    """

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
    """A scoring function resource for evaluating model outputs.
    :param type: The resource type, always scoring_function
    """

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
    data: list[ScoringFn]


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring-functions", method="GET", level=LLAMA_STACK_API_V1)
    async def list_scoring_functions(self) -> ListScoringFunctionsResponse:
        """List all scoring functions.

        :returns: A ListScoringFunctionsResponse.
        """
        ...

    @webmethod(route="/scoring-functions/{scoring_fn_id:path}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_scoring_function(self, scoring_fn_id: str, /) -> ScoringFn:
        """Get a scoring function by its ID.

        :param scoring_fn_id: The ID of the scoring function to get.
        :returns: A ScoringFn.
        """
        ...

    @webmethod(route="/scoring-functions", method="POST", level=LLAMA_STACK_API_V1, deprecated=True)
    async def register_scoring_function(
        self,
        scoring_fn_id: str,
        description: str,
        return_type: ParamType,
        provider_scoring_fn_id: str | None = None,
        provider_id: str | None = None,
        params: ScoringFnParams | None = None,
    ) -> None:
        """Register a scoring function.

        :param scoring_fn_id: The ID of the scoring function to register.
        :param description: The description of the scoring function.
        :param return_type: The return type of the scoring function.
        :param provider_scoring_fn_id: The ID of the provider scoring function to use for the scoring function.
        :param provider_id: The ID of the provider to use for the scoring function.
        :param params: The parameters for the scoring function for benchmark eval, these can be overridden for app eval.
        """
        ...

    @webmethod(
        route="/scoring-functions/{scoring_fn_id:path}", method="DELETE", level=LLAMA_STACK_API_V1, deprecated=True
    )
    async def unregister_scoring_function(self, scoring_fn_id: str) -> None:
        """Unregister a scoring function.

        :param scoring_fn_id: The ID of the scoring function to unregister.
        """
        ...
