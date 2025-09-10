# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.scoring import (
    ScoreBatchResponse,
    ScoreResponse,
    Scoring,
    ScoringResult,
)
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnParams
from llama_stack.core.datatypes import Api
from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate
from llama_stack.providers.utils.common.data_schema_validator import (
    get_valid_schemas,
    validate_dataset_schema,
)

from .config import BasicScoringConfig
from .scoring_fn.docvqa_scoring_fn import DocVQAScoringFn
from .scoring_fn.equality_scoring_fn import EqualityScoringFn
from .scoring_fn.ifeval_scoring_fn import IfEvalScoringFn
from .scoring_fn.regex_parser_math_response_scoring_fn import (
    RegexParserMathResponseScoringFn,
)
from .scoring_fn.regex_parser_scoring_fn import RegexParserScoringFn
from .scoring_fn.subset_of_scoring_fn import SubsetOfScoringFn

FIXED_FNS = [
    EqualityScoringFn,
    SubsetOfScoringFn,
    RegexParserScoringFn,
    RegexParserMathResponseScoringFn,
    IfEvalScoringFn,
    DocVQAScoringFn,
]


class BasicScoringImpl(
    Scoring,
    ScoringFunctionsProtocolPrivate,
):
    def __init__(
        self,
        config: BasicScoringConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.scoring_fn_id_impls = {}

    async def initialize(self) -> None:
        for fn in FIXED_FNS:
            impl = fn()
            for fn_defs in impl.get_supported_scoring_fn_defs():
                self.scoring_fn_id_impls[fn_defs.identifier] = impl

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> list[ScoringFn]:
        scoring_fn_defs_list = [
            fn_def for impl in self.scoring_fn_id_impls.values() for fn_def in impl.get_supported_scoring_fn_defs()
        ]

        for f in scoring_fn_defs_list:
            assert f.identifier.startswith("basic"), "All basic scoring fn must have identifier prefixed with 'basic'! "

        return scoring_fn_defs_list

    async def register_scoring_function(self, function_def: ScoringFn) -> None:
        raise NotImplementedError("Register scoring function not implemented yet")

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: dict[str, ScoringFnParams | None] = None,
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)
        validate_dataset_schema(dataset_def.dataset_schema, get_valid_schemas(Api.scoring.value))

        all_rows = await self.datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=-1,
        )
        res = await self.score(
            input_rows=all_rows.data,
            scoring_functions=scoring_functions,
        )
        if save_results_dataset:
            # TODO: persist and register dataset on to server for reading
            # self.datasets_api.register_dataset()
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res.results,
        )

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_functions: dict[str, ScoringFnParams | None] = None,
    ) -> ScoreResponse:
        res = {}
        for scoring_fn_id in scoring_functions.keys():
            if scoring_fn_id not in self.scoring_fn_id_impls:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")
            scoring_fn = self.scoring_fn_id_impls[scoring_fn_id]
            scoring_fn_params = scoring_functions.get(scoring_fn_id, None)
            score_results = await scoring_fn.score(input_rows, scoring_fn_id, scoring_fn_params)
            agg_results = await scoring_fn.aggregate(score_results, scoring_fn_id, scoring_fn_params)
            res[scoring_fn_id] = ScoringResult(
                score_rows=score_results,
                aggregated_results=agg_results,
            )

        return ScoreResponse(
            results=res,
        )
