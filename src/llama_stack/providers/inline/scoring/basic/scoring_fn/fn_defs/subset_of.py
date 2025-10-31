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

subset_of = ScoringFn(
    identifier="basic::subset_of",
    description="Returns 1.0 if the expected is included in generated, 0.0 otherwise.",
    return_type=NumberType(),
    provider_id="basic",
    provider_resource_id="subset-of",
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.accuracy]),
)
