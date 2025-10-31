# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Any

from llama_stack.apis.scoring import ScoringFnParams, ScoringResultRow
from llama_stack.apis.scoring_functions import ScoringFn
from llama_stack.providers.utils.scoring.aggregation_utils import aggregate_metrics


class BaseScoringFn(ABC):
    """
    Base interface class for Scoring Functions.
    Each scoring function needs to implement the following methods:
    - score_row(self, row)
    - aggregate(self, scoring_fn_results)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def score_row(
        self,
        input_row: dict[str, Any],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> ScoringResultRow:
        raise NotImplementedError()

    @abstractmethod
    async def aggregate(
        self,
        scoring_results: list[ScoringResultRow],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> list[ScoringResultRow]:
        raise NotImplementedError()


class RegisteredBaseScoringFn(BaseScoringFn):
    """
    Interface for native scoring functions that are registered in LlamaStack.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {}

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_supported_scoring_fn_defs(self) -> list[ScoringFn]:
        return list(self.supported_fn_defs_registry.values())

    def register_scoring_fn_def(self, scoring_fn: ScoringFn) -> None:
        if scoring_fn.identifier in self.supported_fn_defs_registry:
            raise ValueError(f"Scoring function def with identifier {scoring_fn.identifier} already exists.")
        self.supported_fn_defs_registry[scoring_fn.identifier] = scoring_fn

    def unregister_scoring_fn_def(self, scoring_fn_id: str) -> None:
        if scoring_fn_id not in self.supported_fn_defs_registry:
            raise ValueError(f"Scoring function def with identifier {scoring_fn_id} does not exist.")
        del self.supported_fn_defs_registry[scoring_fn_id]

    @abstractmethod
    async def score_row(
        self,
        input_row: dict[str, Any],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> ScoringResultRow:
        raise NotImplementedError()

    async def aggregate(
        self,
        scoring_results: list[ScoringResultRow],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> dict[str, Any]:
        params = self.supported_fn_defs_registry[scoring_fn_identifier].params
        if scoring_params is not None:
            if params is None:
                params = scoring_params
            else:
                params.aggregation_functions = scoring_params.aggregation_functions

        aggregation_functions = []
        if params and hasattr(params, "aggregation_functions") and params.aggregation_functions:
            aggregation_functions.extend(params.aggregation_functions)
        return aggregate_metrics(scoring_results, aggregation_functions)

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_fn_identifier: str | None = None,
        scoring_params: ScoringFnParams | None = None,
    ) -> list[ScoringResultRow]:
        return [await self.score_row(input_row, scoring_fn_identifier, scoring_params) for input_row in input_rows]
