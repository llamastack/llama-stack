# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import re
from typing import Any

from llama_stack.apis.inference import Inference, OpenAIChatCompletionRequestWithExtraBody
from llama_stack.apis.scoring import ScoringResultRow
from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.providers.utils.scoring.base_scoring_fn import RegisteredBaseScoringFn

from .fn_defs.llm_as_judge_405b_simpleqa import llm_as_judge_405b_simpleqa
from .fn_defs.llm_as_judge_base import llm_as_judge_base


class LlmAsJudgeScoringFn(RegisteredBaseScoringFn):
    """
    A scoring_fn that assigns
    """

    def __init__(self, inference_api: Inference, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.inference_api = inference_api
        self.supported_fn_defs_registry = {
            llm_as_judge_base.identifier: llm_as_judge_base,
            llm_as_judge_405b_simpleqa.identifier: llm_as_judge_405b_simpleqa,
        }

    async def score_row(
        self,
        input_row: dict[str, Any],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> ScoringResultRow:
        assert scoring_fn_identifier is not None, "Scoring function identifier not found."
        fn_def = self.supported_fn_defs_registry[scoring_fn_identifier]

        # override params if scoring_params is provided
        if scoring_params is not None:
            fn_def.params = scoring_params

        assert fn_def.params is not None, f"LLMAsJudgeparams not found for {fn_def}."
        assert fn_def.params.prompt_template is not None, "LLM Judge prompt_template not found."
        assert fn_def.params.judge_score_regexes is not None, "LLM Judge judge_score_regexes not found."

        input_query = input_row["input_query"]
        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]

        judge_input_msg = fn_def.params.prompt_template.format(
            input_query=input_query,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=fn_def.params.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": judge_input_msg,
                }
            ],
        )
        judge_response = await self.inference_api.openai_chat_completion(params)
        content = judge_response.choices[0].message.content
        rating_regexes = fn_def.params.judge_score_regexes

        judge_rating = None
        for regex in rating_regexes:
            match = re.search(regex, content)
            if match:
                judge_rating = match.group(1)
                break

        return {
            "score": judge_rating,
            "judge_feedback": content,
        }
