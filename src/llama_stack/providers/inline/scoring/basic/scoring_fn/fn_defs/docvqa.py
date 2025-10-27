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

docvqa = ScoringFn(
    identifier="basic::docvqa",
    description="DocVQA Visual Question & Answer scoring function",
    return_type=NumberType(),
    provider_id="basic",
    provider_resource_id="docvqa",
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.accuracy]),
)
