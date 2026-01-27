# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
import threading
from typing import Any

from llama_stack.core.datatypes import Api
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.common.data_schema_validator import (
    get_valid_schemas,
    validate_row_schema,
)
from llama_stack.providers.utils.scoring.aggregation_utils import aggregate_metrics
from llama_stack_api import (
    DatasetIO,
    Datasets,
    IterRowsRequest,
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreRequest,
    ScoreResponse,
    Scoring,
    ScoringFn,
    ScoringFunctionsProtocolPrivate,
    ScoringResult,
    ScoringResultRow,
)

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

# Mapping of scoring function identifiers to their definitions (lightweight, no heavy imports)
SUPPORTED_BRAINTRUST_SCORING_FN_DEFS: dict[str, ScoringFn] = {
    "braintrust::factuality": factuality_fn_def,
    "braintrust::answer-correctness": answer_correctness_fn_def,
    "braintrust::answer-relevancy": answer_relevancy_fn_def,
    "braintrust::answer-similarity": answer_similarity_fn_def,
    "braintrust::faithfulness": faithfulness_fn_def,
    "braintrust::context-entity-recall": context_entity_recall_fn_def,
    "braintrust::context-precision": context_precision_fn_def,
    "braintrust::context-recall": context_recall_fn_def,
    "braintrust::context-relevancy": context_relevancy_fn_def,
}

# Lazy-loaded evaluators (defers loading autoevals and its pyarrow dependency)
_braintrust_evaluators: dict[str, Any] | None = None
_braintrust_evaluators_lock = threading.Lock()


def _get_braintrust_evaluators() -> dict[str, Any]:
    """Lazily load autoevals evaluators on first use.

    This defers importing autoevals (and its pyarrow dependency) until
    braintrust scoring is actually needed, saving ~63MB of memory at startup.
    """
    global _braintrust_evaluators
    if _braintrust_evaluators is not None:
        return _braintrust_evaluators

    with _braintrust_evaluators_lock:
        if _braintrust_evaluators is not None:
            return _braintrust_evaluators

        # Import autoevals only when evaluators are actually needed
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

        _braintrust_evaluators = {
            "braintrust::factuality": Factuality(),
            "braintrust::answer-correctness": AnswerCorrectness(),
            "braintrust::answer-relevancy": AnswerRelevancy(),
            "braintrust::answer-similarity": AnswerSimilarity(),
            "braintrust::faithfulness": Faithfulness(),
            "braintrust::context-entity-recall": ContextEntityRecall(),
            "braintrust::context-precision": ContextPrecision(),
            "braintrust::context-recall": ContextRecall(),
            "braintrust::context-relevancy": ContextRelevancy(),
        }
        return _braintrust_evaluators


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
        # Use the lightweight fn_defs dict (no heavy autoevals import)
        self.supported_fn_defs_registry = SUPPORTED_BRAINTRUST_SCORING_FN_DEFS

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
        request: ScoreBatchRequest,
    ) -> ScoreBatchResponse:
        await self.set_api_key()

        all_rows = await self.datasetio_api.iterrows(IterRowsRequest(dataset_id=request.dataset_id, limit=-1))
        score_request = ScoreRequest(
            input_rows=all_rows.data,
            scoring_functions=request.scoring_functions,
        )
        res = await self.score(score_request)
        if request.save_results_dataset:
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
        # Lazy-load evaluators only when scoring is actually performed
        evaluators = _get_braintrust_evaluators()
        evaluator = evaluators[scoring_fn_identifier]

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
        request: ScoreRequest,
    ) -> ScoreResponse:
        await self.set_api_key()
        res = {}
        for scoring_fn_id in request.scoring_functions:
            if scoring_fn_id not in self.supported_fn_defs_registry:
                raise ValueError(f"Scoring function {scoring_fn_id} is not supported.")

            score_results = [await self.score_row(input_row, scoring_fn_id) for input_row in request.input_rows]
            aggregation_functions = self.supported_fn_defs_registry[scoring_fn_id].params.aggregation_functions

            # override scoring_fn params if provided
            if request.scoring_functions[scoring_fn_id] is not None:
                override_params = request.scoring_functions[scoring_fn_id]
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
