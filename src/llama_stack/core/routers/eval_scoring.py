# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.log import get_logger
from llama_stack_api import (
    Eval,
    EvaluateResponse,
    EvaluateRowsRequest,
    Job,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    RoutingTable,
    RunEvalRequest,
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreRequest,
    ScoreResponse,
    Scoring,
)

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
        request: ScoreBatchRequest,
    ) -> ScoreBatchResponse:
        logger.debug(f"ScoringRouter.score_batch: {request.dataset_id}")
        res = {}
        for fn_identifier in request.scoring_functions.keys():
            provider = await self.routing_table.get_provider_impl(fn_identifier)
            # Create a request for this specific scoring function
            single_fn_request = ScoreBatchRequest(
                dataset_id=request.dataset_id,
                scoring_functions={fn_identifier: request.scoring_functions[fn_identifier]},
                save_results_dataset=request.save_results_dataset,
            )
            score_response = await provider.score_batch(single_fn_request)
            res.update(score_response.results)

        if request.save_results_dataset:
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res,
        )

    async def score(
        self,
        request: ScoreRequest,
    ) -> ScoreResponse:
        logger.debug(f"ScoringRouter.score: {len(request.input_rows)} rows, {len(request.scoring_functions)} functions")
        res = {}
        # look up and map each scoring function to its provider impl
        for fn_identifier in request.scoring_functions.keys():
            provider = await self.routing_table.get_provider_impl(fn_identifier)
            # Create a request for this specific scoring function
            single_fn_request = ScoreRequest(
                input_rows=request.input_rows,
                scoring_functions={fn_identifier: request.scoring_functions[fn_identifier]},
            )
            score_response = await provider.score(single_fn_request)
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
        request: RunEvalRequest,
    ) -> Job:
        logger.debug(f"EvalRouter.run_eval: {request.benchmark_id}")
        provider = await self.routing_table.get_provider_impl(request.benchmark_id)
        return await provider.run_eval(request)

    async def evaluate_rows(
        self,
        request: EvaluateRowsRequest,
    ) -> EvaluateResponse:
        logger.debug(f"EvalRouter.evaluate_rows: {request.benchmark_id}, {len(request.input_rows)} rows")
        provider = await self.routing_table.get_provider_impl(request.benchmark_id)
        return await provider.evaluate_rows(request)

    async def job_status(
        self,
        request: JobStatusRequest,
    ) -> Job:
        logger.debug(f"EvalRouter.job_status: {request.benchmark_id}, {request.job_id}")
        provider = await self.routing_table.get_provider_impl(request.benchmark_id)
        return await provider.job_status(request)

    async def job_cancel(
        self,
        request: JobCancelRequest,
    ) -> None:
        logger.debug(f"EvalRouter.job_cancel: {request.benchmark_id}, {request.job_id}")
        provider = await self.routing_table.get_provider_impl(request.benchmark_id)
        await provider.job_cancel(request)

    async def job_result(
        self,
        request: JobResultRequest,
    ) -> EvaluateResponse:
        logger.debug(f"EvalRouter.job_result: {request.benchmark_id}, {request.job_id}")
        provider = await self.routing_table.get_provider_impl(request.benchmark_id)
        return await provider.job_result(request)
