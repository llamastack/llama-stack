# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import re
from typing import Any

from llama_stack.apis.scoring import ScoringResultRow
from llama_stack.apis.scoring_functions import ScoringFnParams, ScoringFnParamsType
from llama_stack.providers.utils.scoring.base_scoring_fn import RegisteredBaseScoringFn

from .fn_defs.regex_parser_multiple_choice_answer import (
    regex_parser_multiple_choice_answer,
)


class RegexParserScoringFn(RegisteredBaseScoringFn):
    """
    A scoring_fn that parses answer from generated response according to context and check match with expected_answer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {
            regex_parser_multiple_choice_answer.identifier: regex_parser_multiple_choice_answer,
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

        # parse answer according to regex
        parsed_answer = None
        for regex in fn_def.params.parsing_regexes:
            match = re.search(regex, generated_answer)
            if match:
                parsed_answer = match.group(1)
                break

        score = 1.0 if parsed_answer and parsed_answer == expected_answer else 0.0
        return {
            "score": score,
        }
