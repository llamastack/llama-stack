# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnParams

from .models import ScoreBatchResponse, ScoreResponse


class ScoringFunctionStore(Protocol):
    def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn: ...


@runtime_checkable
class ScoringService(Protocol):
    scoring_function_store: ScoringFunctionStore

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: dict[str, ScoringFnParams | None],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        """Score a batch of rows."""
        ...

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_functions: dict[str, ScoringFnParams | None],
    ) -> ScoreResponse:
        """Score a list of rows."""
        ...
