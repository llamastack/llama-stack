# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.apis.scoring import ScoringResultRow
from llama_stack.apis.scoring_functions import ScoringFnParams, ScoringFnParamsType
from llama_stack.providers.utils.scoring.base_scoring_fn import RegisteredBaseScoringFn

from ..utils.math_utils import first_answer, normalize_final_answer, try_evaluate_frac, try_evaluate_latex
from .fn_defs.regex_parser_math_response import (
    regex_parser_math_response,
)


class RegexParserMathResponseScoringFn(RegisteredBaseScoringFn):
    """
    A scoring_fn for math benchamrks that parses answer from generated response according to context and check match with expected_answer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {
            regex_parser_math_response.identifier: regex_parser_math_response,
        }

    async def score_row(
        self,
        input_row: dict[str, Any],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> ScoringResultRow:
        assert scoring_fn_identifier is not None, "Scoring function identifier not found."
        fn_def = self.supported_fn_defs_registry[scoring_fn_identifier]
        if scoring_params is not None:
            fn_def.params = scoring_params

        assert fn_def.params is not None and fn_def.params.type == ScoringFnParamsType.regex_parser.value, (
            f"RegexParserScoringFnParams not found for {fn_def}."
        )

        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]

        parsing_regexes = fn_def.params.parsing_regexes
        assert len(parsing_regexes) == 1, (
            "Only one parsing regex is supported for regex_parser_math_response scoring function."
        )
        parsing_regexes = fn_def.params.parsing_regexes[0]

        normalized_generated_answer = normalize_final_answer(
            first_answer(generated_answer),
            parsing_regexes,
            match_first=True,
        )
        normalized_generated_answer = try_evaluate_frac(try_evaluate_latex(normalized_generated_answer))

        normalized_expected_answer = normalize_final_answer(expected_answer, r".*")
        normalized_expected_answer = try_evaluate_frac(try_evaluate_latex(normalized_expected_answer))

        score = 1.0 if normalized_generated_answer == normalized_expected_answer else 0.0
        return {
            "score": score,
        }
