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

context_relevancy_fn_def = ScoringFn(
    identifier="braintrust::context-relevancy",
    description=(
        "Assesses how relevant the provided context is to the given question. See: github.com/braintrustdata/autoevals"
    ),
    provider_id="braintrust",
    provider_resource_id="context-relevancy",
    return_type=NumberType(),
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.average]),
)
