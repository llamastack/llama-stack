# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

import requests

from llama_stack.apis.agents import Agents
from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference
from llama_stack.apis.scoring import Scoring, ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

from .....apis.common.job_types import Job, JobStatus
from .....apis.eval.eval import BenchmarkConfig, Eval, EvaluateResponse
from .config import NVIDIAEvalConfig

DEFAULT_NAMESPACE = "nvidia"


class NVIDIAEvalImpl(
    Eval,
    BenchmarksProtocolPrivate,
    ModelRegistryHelper,
):
    def __init__(
        self,
        config: NVIDIAEvalConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        scoring_api: Scoring,
        inference_api: Inference,
        agents_api: Agents,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.scoring_api = scoring_api
        self.inference_api = inference_api
        self.agents_api = agents_api

        ModelRegistryHelper.__init__(self)

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def _evaluator_get(self, path: str):
        """Helper for making GET requests to the evaluator service."""
        response = requests.get(url=f"{self.config.evaluator_url}{path}")
        response.raise_for_status()
        return response.json()

    async def _evaluator_post(self, path: str, data: dict[str, Any]):
        """Helper for making POST requests to the evaluator service."""
        response = requests.post(url=f"{self.config.evaluator_url}{path}", json=data)
        response.raise_for_status()
        return response.json()

    async def _evaluator_delete(self, path: str) -> None:
        """Helper for making DELETE requests to the evaluator service."""
        response = requests.delete(url=f"{self.config.evaluator_url}{path}")
        response.raise_for_status()

    async def register_benchmark(self, task_def: Benchmark) -> None:
        """Register a benchmark as an evaluation configuration."""
        await self._evaluator_post(
            "/v1/evaluation/configs",
            {
                "namespace": DEFAULT_NAMESPACE,
                "name": task_def.benchmark_id,
                # metadata is copied to request body as-is
                **task_def.metadata,
            },
        )

    async def unregister_benchmark(self, benchmark_id: str) -> None:
        """Unregister a benchmark evaluation configuration from NeMo Evaluator."""
        await self._evaluator_delete(f"/v1/evaluation/configs/{DEFAULT_NAMESPACE}/{benchmark_id}")

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        """Run an evaluation job for a benchmark."""
        model = (
            benchmark_config.eval_candidate.model
            if benchmark_config.eval_candidate.type == "model"
            else benchmark_config.eval_candidate.config.model
        )
        nvidia_model = self.get_provider_model_id(model) or model

        result = await self._evaluator_post(
            "/v1/evaluation/jobs",
            {
                "config": f"{DEFAULT_NAMESPACE}/{benchmark_id}",
                "target": {"type": "model", "model": nvidia_model},
            },
        )

        return Job(job_id=result["id"], status=JobStatus.in_progress)

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        raise NotImplementedError()

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        """Get the status of an evaluation job.

        EvaluatorStatus: "created", "pending", "running", "cancelled", "cancelling", "failed", "completed".
        JobStatus: "scheduled", "in_progress", "completed", "cancelled", "failed"
        """
        result = await self._evaluator_get(f"/v1/evaluation/jobs/{job_id}")
        result_status = result["status"]

        job_status = JobStatus.failed
        if result_status in ["created", "pending"]:
            job_status = JobStatus.scheduled
        elif result_status in ["running"]:
            job_status = JobStatus.in_progress
        elif result_status in ["completed"]:
            job_status = JobStatus.completed
        elif result_status in ["cancelled"]:
            job_status = JobStatus.cancelled

        return Job(job_id=job_id, status=job_status)

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel the evaluation job."""
        await self._evaluator_post(f"/v1/evaluation/jobs/{job_id}/cancel", {})

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Returns the results of the evaluation job."""

        job = await self.job_status(benchmark_id, job_id)
        status = job.status
        if not status or status != JobStatus.completed:
            raise ValueError(f"Job {job_id} not completed. Status: {status.value}")

        result = await self._evaluator_get(f"/v1/evaluation/jobs/{job_id}/results")

        return EvaluateResponse(
            # TODO: these are stored in detailed results on NeMo Evaluator side; can be added
            generations=[],
            scores={
                benchmark_id: ScoringResult(
                    score_rows=[],
                    aggregated_results=result,
                )
            },
        )
