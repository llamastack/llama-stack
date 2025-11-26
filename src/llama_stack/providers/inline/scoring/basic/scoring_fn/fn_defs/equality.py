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

equality = ScoringFn(
    identifier="basic::equality",
    description="Returns 1.0 if the input is equal to the target, 0.0 otherwise.",
    provider_id="basic",
    provider_resource_id="equality",
    return_type=NumberType(),
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.accuracy]),
)
