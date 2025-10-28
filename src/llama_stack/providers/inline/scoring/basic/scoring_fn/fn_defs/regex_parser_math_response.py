# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.scoring_functions import (
    AggregationFunctionType,
    RegexParserScoringFnParams,
    ScoringFn,
)

MATH_ANSWER_REGEXES = [r".*final answer is:?\s*\$\\boxed{(?P<X>.*)}\$"]


regex_parser_math_response = ScoringFn(
    identifier="basic::regex_parser_math_response",
    description="For math related benchmarks, extract answer from the generated response and expected_answer and see if they match",
    return_type=NumberType(),
    provider_id="basic",
    provider_resource_id="regex-parser-math-response",
    params=RegexParserScoringFnParams(
        parsing_regexes=MATH_ANSWER_REGEXES,
        aggregation_functions=[AggregationFunctionType.accuracy],
    ),
)
