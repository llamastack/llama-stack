# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.scoring_functions import (
    AggregationFunctionType,
    BasicScoringFnParams,
    ScoringFn,
)

answer_correctness_fn_def = ScoringFn(
    identifier="braintrust::answer-correctness",
    description=(
        "Scores the correctness of the answer based on the ground truth. "
        "Uses Braintrust LLM-based scorer from autoevals library."
    ),
    provider_id="braintrust",
    provider_resource_id="answer-correctness",
    return_type=NumberType(),
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.average]),
)
