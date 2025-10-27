# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from typing import Any

from autoevals.llm import Factuality
from autoevals.ragas import (
    AnswerCorrectness,
    AnswerRelevancy,
    AnswerSimilarity,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextRelevancy,
    Faithfulness,
)
from pydantic import BaseModel

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.scoring import (
    ScoreBatchResponse,
    ScoreResponse,
    Scoring,
    ScoringResult,
    ScoringResultRow,
)
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnParams
from llama_stack.core.datatypes import Api
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ScoringFunctionsProtocolPrivate
from llama_stack.providers.utils.common.data_schema_validator import (
    get_valid_schemas,
    validate_dataset_schema,
    validate_row_schema,
)
from llama_stack.providers.utils.scoring.aggregation_utils import aggregate_metrics

from .config import BraintrustScoringConfig
from .scoring_fn.fn_defs.answer_correctness import answer_correctness_fn_def
from .scoring_fn.fn_defs.answer_relevancy import answer_relevancy_fn_def
from .scoring_fn.fn_defs.answer_similarity import answer_similarity_fn_def
from .scoring_fn.fn_defs.context_entity_recall import context_entity_recall_fn_def
from .scoring_fn.fn_defs.context_precision import context_precision_fn_def
from .scoring_fn.fn_defs.context_recall import context_recall_fn_def
from .scoring_fn.fn_defs.context_relevancy import context_relevancy_fn_def
from .scoring_fn.fn_defs.factuality import factuality_fn_def
from .scoring_fn.fn_defs.faithfulness import faithfulness_fn_def


class BraintrustScoringFnEntry(BaseModel):
    identifier: str
    evaluator: Any
    fn_def: ScoringFn


SUPPORTED_BRAINTRUST_SCORING_FN_ENTRY = [
    BraintrustScoringFnEntry(
        identifier="braintrust::factuality",
        evaluator=Factuality(),
        fn_def=factuality_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::answer-correctness",
        evaluator=AnswerCorrectness(),
        fn_def=answer_correctness_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::answer-relevancy",
        evaluator=AnswerRelevancy(),
        fn_def=answer_relevancy_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::answer-similarity",
        evaluator=AnswerSimilarity(),
        fn_def=answer_similarity_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::faithfulness",
        evaluator=Faithfulness(),
        fn_def=faithfulness_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::context-entity-recall",
        evaluator=ContextEntityRecall(),
        fn_def=context_entity_recall_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::context-precision",
        evaluator=ContextPrecision(),
        fn_def=context_precision_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::context-recall",
        evaluator=ContextRecall(),
        fn_def=context_recall_fn_def,
    ),
    BraintrustScoringFnEntry(
        identifier="braintrust::context-relevancy",
        evaluator=ContextRelevancy(),
        fn_def=context_relevancy_fn_def,
    ),
]


class BraintrustScoringImpl(
    Scoring,
    ScoringFunctionsProtocolPrivate,
    NeedsRequestProviderData,
):
    def __init__(
        self,
        config: BraintrustScoringConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

        self.braintrust_evaluators = {
            entry.identifier: entry.evaluator for entry in SUPPORTED_BRAINTRUST_SCORING_FN_ENTRY
        }
        self.supported_fn_defs_registry = {
            entry.identifier: entry.fn_def for entry in SUPPORTED_BRAINTRUST_SCORING_FN_ENTRY
        }

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def list_scoring_functions(self) -> list[ScoringFn]:
        scoring_fn_defs_list = list(self.supported_fn_defs_registry.values())
        for f in scoring_fn_defs_list:
            assert f.identifier.startswith("braintrust"), (
                "All braintrust scoring fn must have identifier prefixed with 'braintrust'! "
            )

        return scoring_fn_defs_list

    async def register_scoring_function(self, scoring_fn: ScoringFn) -> None:
        raise NotImplementedError("Registering scoring function not allowed for braintrust provider")

    async def set_api_key(self) -> None:
        # api key is in the request headers
        if not self.config.openai_api_key:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.openai_api_key:
                raise ValueError(
                    'Pass OpenAI API Key in the header X-LlamaStack-Provider-Data as { "openai_api_key": <your api key>}'
                )
            self.config.openai_api_key = provider_data.openai_api_key

        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: dict[str, ScoringFnParams | None],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        await self.set_api_key()

        dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)
        validate_dataset_schema(dataset_def.dataset_schema, get_valid_schemas(Api.scoring.value))

        all_rows = await self.datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=-1,
        )
        res = await self.score(input_rows=all_rows.data, scoring_functions=scoring_functions)
        if save_results_dataset:
            # TODO: persist and register dataset on to server for reading
            # self.datasets_api.register_dataset()
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res.results,
        )

    async def score_row(self, input_row: dict[str, Any], scoring_fn_identifier: str | None = None) -> ScoringResultRow:
        validate_row_schema(input_row, get_valid_schemas(Api.scoring.value))
        await self.set_api_key()
        assert scoring_fn_identifier is not None, "scoring_fn_identifier cannot be None"
        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]
        input_query = input_row["input_query"]
        evaluator = self.braintrust_evaluators[scoring_fn_identifier]

        result = evaluator(
            generated_answer,
            expected_answer,
            input=input_query,
            context=input_row["context"] if "context" in input_row else None,
        )
        score = result.score
        return {"score": score, "metadata": result.metadata}

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_functions: dict[str, ScoringFnParams | None],
    ) -> ScoreResponse:
        await self.set_api_key()
        res = {}
        for scoring_fn_id in scoring_functions:
            if scoring_fn_id not in self.supported_fn_defs_registry:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")

            score_results = [await self.score_row(input_row, scoring_fn_id) for input_row in input_rows]
            aggregation_functions = self.supported_fn_defs_registry[scoring_fn_id].params.aggregation_functions

            # override scoring_fn params if provided
            if scoring_functions[scoring_fn_id] is not None:
                override_params = scoring_functions[scoring_fn_id]
                if override_params.aggregation_functions:
                    aggregation_functions = override_params.aggregation_functions

            agg_results = aggregate_metrics(score_results, aggregation_functions)
            res[scoring_fn_id] = ScoringResult(
                score_rows=score_results,
                aggregated_results=agg_results,
            )

        return ScoreResponse(
            results=res,
        )
