# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.log import get_logger
from llama_stack_api import (
    BenchmarkConfig,
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
    resolve_evaluate_rows_request,
    resolve_job_cancel_request,
    resolve_job_result_request,
    resolve_job_status_request,
    resolve_run_eval_request,
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
        request: RunEvalRequest | None = None,
        *,
        benchmark_id: str | None = None,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> Job:
        """Run an evaluation on a benchmark.

        Supports both new-style (request object) and old-style (individual parameters).
        Old-style usage is deprecated and will emit a DeprecationWarning.

        Args:
            request: The new-style request object (preferred)
            benchmark_id: (Deprecated) The benchmark ID
            benchmark_config: (Deprecated) The benchmark configuration

        Returns:
            Job object representing the evaluation job
        """
        resolved_request = resolve_run_eval_request(
            request, benchmark_id=benchmark_id, benchmark_config=benchmark_config
        )
        logger.debug(f"EvalRouter.run_eval: {resolved_request.benchmark_id}")
        provider = await self.routing_table.get_provider_impl(resolved_request.benchmark_id)
        return await provider.run_eval(resolved_request)

    async def evaluate_rows(
        self,
        request: EvaluateRowsRequest | None = None,
        *,
        benchmark_id: str | None = None,
        input_rows: list[dict[str, Any]] | None = None,
        scoring_functions: list[str] | None = None,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark.

        Supports both new-style (request object) and old-style (individual parameters).
        Old-style usage is deprecated and will emit a DeprecationWarning.

        Args:
            request: The new-style request object (preferred)
            benchmark_id: (Deprecated) The benchmark ID
            input_rows: (Deprecated) The rows to evaluate
            scoring_functions: (Deprecated) The scoring functions to use
            benchmark_config: (Deprecated) The benchmark configuration

        Returns:
            EvaluateResponse object containing generations and scores
        """
        resolved_request = resolve_evaluate_rows_request(
            request,
            benchmark_id=benchmark_id,
            input_rows=input_rows,
            scoring_functions=scoring_functions,
            benchmark_config=benchmark_config,
        )
        logger.debug(
            f"EvalRouter.evaluate_rows: {resolved_request.benchmark_id}, {len(resolved_request.input_rows)} rows"
        )
        provider = await self.routing_table.get_provider_impl(resolved_request.benchmark_id)
        return await provider.evaluate_rows(resolved_request)

    async def job_status(
        self,
        request: JobStatusRequest | None = None,
        *,
        benchmark_id: str | None = None,
        job_id: str | None = None,
    ) -> Job:
        """Get the status of a job.

        Supports both new-style (request object) and old-style (individual parameters).
        Old-style usage is deprecated and will emit a DeprecationWarning.

        Args:
            request: The new-style request object (preferred)
            benchmark_id: (Deprecated) The benchmark ID
            job_id: (Deprecated) The job ID

        Returns:
            Job object with the current status
        """
        resolved_request = resolve_job_status_request(request, benchmark_id=benchmark_id, job_id=job_id)
        logger.debug(f"EvalRouter.job_status: {resolved_request.benchmark_id}, {resolved_request.job_id}")
        provider = await self.routing_table.get_provider_impl(resolved_request.benchmark_id)
        return await provider.job_status(resolved_request)

    async def job_cancel(
        self,
        request: JobCancelRequest | None = None,
        *,
        benchmark_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        """Cancel a job.

        Supports both new-style (request object) and old-style (individual parameters).
        Old-style usage is deprecated and will emit a DeprecationWarning.

        Args:
            request: The new-style request object (preferred)
            benchmark_id: (Deprecated) The benchmark ID
            job_id: (Deprecated) The job ID

        Returns:
            None
        """
        resolved_request = resolve_job_cancel_request(request, benchmark_id=benchmark_id, job_id=job_id)
        logger.debug(f"EvalRouter.job_cancel: {resolved_request.benchmark_id}, {resolved_request.job_id}")
        provider = await self.routing_table.get_provider_impl(resolved_request.benchmark_id)
        await provider.job_cancel(resolved_request)

    async def job_result(
        self,
        request: JobResultRequest | None = None,
        *,
        benchmark_id: str | None = None,
        job_id: str | None = None,
    ) -> EvaluateResponse:
        """Get the result of a job.

        Supports both new-style (request object) and old-style (individual parameters).
        Old-style usage is deprecated and will emit a DeprecationWarning.

        Args:
            request: The new-style request object (preferred)
            benchmark_id: (Deprecated) The benchmark ID
            job_id: (Deprecated) The job ID

        Returns:
            EvaluateResponse object with the job results
        """
        resolved_request = resolve_job_result_request(request, benchmark_id=benchmark_id, job_id=job_id)
        logger.debug(f"EvalRouter.job_result: {resolved_request.benchmark_id}, {resolved_request.job_id}")
        provider = await self.routing_table.get_provider_impl(resolved_request.benchmark_id)
        return await provider.job_result(resolved_request)
