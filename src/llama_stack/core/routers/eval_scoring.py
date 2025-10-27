# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse, Job
from llama_stack.apis.scoring import (
    ScoreBatchResponse,
    ScoreResponse,
    Scoring,
    ScoringFnParams,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core::routers")


class ScoringRouter(Scoring):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing ScoringRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("ScoringRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("ScoringRouter.shutdown")
        pass

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: dict[str, ScoringFnParams | None] = None,
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        logger.debug(f"ScoringRouter.score_batch: {dataset_id}")
        res = {}
        for fn_identifier in scoring_functions.keys():
            provider = await self.routing_table.get_provider_impl(fn_identifier)
            score_response = await provider.score_batch(
                dataset_id=dataset_id,
                scoring_functions={fn_identifier: scoring_functions[fn_identifier]},
            )
            res.update(score_response.results)

        if save_results_dataset:
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res,
        )

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_functions: dict[str, ScoringFnParams | None] = None,
    ) -> ScoreResponse:
        logger.debug(f"ScoringRouter.score: {len(input_rows)} rows, {len(scoring_functions)} functions")
        res = {}
        # look up and map each scoring function to its provider impl
        for fn_identifier in scoring_functions.keys():
            provider = await self.routing_table.get_provider_impl(fn_identifier)
            score_response = await provider.score(
                input_rows=input_rows,
                scoring_functions={fn_identifier: scoring_functions[fn_identifier]},
            )
            res.update(score_response.results)

        return ScoreResponse(results=res)


class EvalRouter(Eval):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing EvalRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("EvalRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("EvalRouter.shutdown")
        pass

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        logger.debug(f"EvalRouter.run_eval: {benchmark_id}")
        provider = await self.routing_table.get_provider_impl(benchmark_id)
        return await provider.run_eval(
            benchmark_id=benchmark_id,
            benchmark_config=benchmark_config,
        )

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        logger.debug(f"EvalRouter.evaluate_rows: {benchmark_id}, {len(input_rows)} rows")
        provider = await self.routing_table.get_provider_impl(benchmark_id)
        return await provider.evaluate_rows(
            benchmark_id=benchmark_id,
            input_rows=input_rows,
            scoring_functions=scoring_functions,
            benchmark_config=benchmark_config,
        )

    async def job_status(
        self,
        benchmark_id: str,
        job_id: str,
    ) -> Job:
        logger.debug(f"EvalRouter.job_status: {benchmark_id}, {job_id}")
        provider = await self.routing_table.get_provider_impl(benchmark_id)
        return await provider.job_status(benchmark_id, job_id)

    async def job_cancel(
        self,
        benchmark_id: str,
        job_id: str,
    ) -> None:
        logger.debug(f"EvalRouter.job_cancel: {benchmark_id}, {job_id}")
        provider = await self.routing_table.get_provider_impl(benchmark_id)
        await provider.job_cancel(
            benchmark_id,
            job_id,
        )

    async def job_result(
        self,
        benchmark_id: str,
        job_id: str,
    ) -> EvaluateResponse:
        logger.debug(f"EvalRouter.job_result: {benchmark_id}, {job_id}")
        provider = await self.routing_table.get_provider_impl(benchmark_id)
        return await provider.job_result(
            benchmark_id,
            job_id,
        )
