# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference
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

from .config import LlmAsJudgeScoringConfig
from .scoring_fn.llm_as_judge_scoring_fn import LlmAsJudgeScoringFn

LLM_JUDGE_FN = LlmAsJudgeScoringFn


class LlmAsJudgeScoringImpl(
    Scoring,
    ScoringFunctionsProtocolPrivate,
):
    def __init__(
        self,
        config: LlmAsJudgeScoringConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        inference_api: Inference,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.inference_api = inference_api

    async def initialize(self) -> None:
        impl = LLM_JUDGE_FN(inference_api=self.inference_api)
        self.llm_as_judge_fn = impl

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> list[ScoringFn]:
        scoring_fn_defs_list = self.llm_as_judge_fn.get_supported_scoring_fn_defs()

        for f in self.llm_as_judge_fn.get_supported_scoring_fn_defs():
            assert f.identifier.startswith("llm-as-judge"), (
                "All llm-as-judge scoring fn must have identifier prefixed with 'llm-as-judge'! "
            )

        return scoring_fn_defs_list

    async def register_scoring_function(self, function_def: ScoringFn) -> None:
        self.llm_as_judge_fn.register_scoring_fn_def(function_def)

    async def unregister_scoring_function(self, scoring_fn_id: str) -> None:
        self.llm_as_judge_fn.unregister_scoring_fn_def(scoring_fn_id)

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
            scoring_fn = self.llm_as_judge_fn
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
