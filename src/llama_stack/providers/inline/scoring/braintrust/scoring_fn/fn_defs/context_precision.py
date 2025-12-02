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

context_precision_fn_def = ScoringFn(
    identifier="braintrust::context-precision",
    description=(
        "Measures how much of the provided context is actually relevant to answering the "
        "question. See: github.com/braintrustdata/autoevals"
    ),
    provider_id="braintrust",
    provider_resource_id="context-precision",
    return_type=NumberType(),
    params=BasicScoringFnParams(aggregation_functions=[AggregationFunctionType.average]),
)
