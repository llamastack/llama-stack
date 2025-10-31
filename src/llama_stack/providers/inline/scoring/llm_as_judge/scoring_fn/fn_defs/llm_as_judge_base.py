# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.scoring_functions import LLMAsJudgeScoringFnParams, ScoringFn

llm_as_judge_base = ScoringFn(
    identifier="llm-as-judge::base",
    description="Llm As Judge Scoring Function",
    return_type=NumberType(),
    provider_id="llm-as-judge",
    provider_resource_id="llm-as-judge-base",
    params=LLMAsJudgeScoringFnParams(
        judge_model="meta-llama/Llama-3.1-405B-Instruct",
        prompt_template="Enter custom LLM as Judge Prompt Template",
    ),
)
