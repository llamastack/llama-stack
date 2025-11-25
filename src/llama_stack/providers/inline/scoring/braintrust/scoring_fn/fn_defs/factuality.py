# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import (
    AggregationFunctionType,
    BasicScoringFnParams,
    NumberType,
    ScoringFn,
)

factuality_fn_def = ScoringFn(
    identifier="braintrust::factuality",
    description=(
        "Test output factuality against expected value using Braintrust LLM scorer. "
        "See: github.com/braintrustdata/autoevals"
    ),
    provider_id="braintrust",
    provider_resource_id="factuality",
    return_type=NumberType(),
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.average]),
)
