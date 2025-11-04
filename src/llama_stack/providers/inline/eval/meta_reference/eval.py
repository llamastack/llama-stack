# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from typing import Any

from tqdm import tqdm

from llama_stack.apis.agents import Agents
from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import (
    Inference,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.scoring import Scoring
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.providers.utils.common.data_schema_validator import ColumnName
from llama_stack.providers.utils.kvstore import kvstore_impl

from .....apis.common.job_types import Job, JobStatus
from .....apis.eval.eval import BenchmarkConfig, Eval, EvaluateResponse
from .config import MetaReferenceEvalConfig

EVAL_TASKS_PREFIX = "benchmarks:"


class MetaReferenceEvalImpl(
    Eval,
    BenchmarksProtocolPrivate,
):
    def __init__(
        self,
        config: MetaReferenceEvalConfig,
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

        # TODO: assume sync job, will need jobs API for async scheduling
        self.jobs = {}

        self.benchmarks = {}

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing benchmarks from kvstore
        start_key = EVAL_TASKS_PREFIX
        end_key = f"{EVAL_TASKS_PREFIX}\xff"
        stored_benchmarks = await self.kvstore.values_in_range(start_key, end_key)

        for benchmark in stored_benchmarks:
            benchmark = Benchmark.model_validate_json(benchmark)
            self.benchmarks[benchmark.identifier] = benchmark

    async def shutdown(self) -> None: ...

    async def register_benchmark(self, task_def: Benchmark) -> None:
        # Store in kvstore
        key = f"{EVAL_TASKS_PREFIX}{task_def.identifier}"
        await self.kvstore.set(
            key=key,
            value=task_def.model_dump_json(),
        )
        self.benchmarks[task_def.identifier] = task_def

    async def unregister_benchmark(self, benchmark_id: str) -> None:
        if benchmark_id in self.benchmarks:
            del self.benchmarks[benchmark_id]

        key = f"{EVAL_TASKS_PREFIX}{benchmark_id}"
        await self.kvstore.delete(key)

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        task_def = self.benchmarks[benchmark_id]
        dataset_id = task_def.dataset_id
        scoring_functions = task_def.scoring_functions

        # TODO (xiyan): validate dataset schema
        # dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)

        all_rows = await self.datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=(-1 if benchmark_config.num_examples is None else benchmark_config.num_examples),
        )
        res = await self.evaluate_rows(
            benchmark_id=benchmark_id,
            input_rows=all_rows.data,
            scoring_functions=scoring_functions,
            benchmark_config=benchmark_config,
        )

        # TODO: currently needs to wait for generation before returning
        # need job scheduler queue (ray/celery) w/ jobs api
        job_id = str(len(self.jobs))
        self.jobs[job_id] = res
        return Job(job_id=job_id, status=JobStatus.completed)

    async def _run_model_generation(
        self, input_rows: list[dict[str, Any]], benchmark_config: BenchmarkConfig
    ) -> list[dict[str, Any]]:
        candidate = benchmark_config.eval_candidate
        assert candidate.sampling_params.max_tokens is not None, "SamplingParams.max_tokens must be provided"
        sampling_params = {"max_tokens": candidate.sampling_params.max_tokens}

        generations = []
        for x in tqdm(input_rows):
            if ColumnName.completion_input.value in x:
                if candidate.sampling_params.stop:
                    sampling_params["stop"] = candidate.sampling_params.stop

                input_content = json.loads(x[ColumnName.completion_input.value])
                params = OpenAICompletionRequestWithExtraBody(
                    model=candidate.model,
                    prompt=input_content,
                    **sampling_params,
                )
                response = await self.inference_api.openai_completion(params)
                generations.append({ColumnName.generated_answer.value: response.choices[0].text})
            elif ColumnName.chat_completion_input.value in x:
                chat_completion_input_json = json.loads(x[ColumnName.chat_completion_input.value])
                input_messages = [
                    OpenAIUserMessageParam(**x) for x in chat_completion_input_json if x["role"] == "user"
                ]

                messages = []
                if candidate.system_message:
                    messages.append(candidate.system_message)

                messages += [OpenAISystemMessageParam(**x) for x in chat_completion_input_json if x["role"] == "system"]

                messages += input_messages
                params = OpenAIChatCompletionRequestWithExtraBody(
                    model=candidate.model,
                    messages=messages,
                    **sampling_params,
                )
                response = await self.inference_api.openai_chat_completion(params)
                generations.append({ColumnName.generated_answer.value: response.choices[0].message.content})
            else:
                raise ValueError("Invalid input row")

        return generations

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        candidate = benchmark_config.eval_candidate
        # Agent evaluation removed
        if candidate.type == "model":
            generations = await self._run_model_generation(input_rows, benchmark_config)
        else:
            raise ValueError(f"Invalid candidate type: {candidate.type}")

        # scoring with generated_answer
        score_input_rows = [
            input_r | generated_r for input_r, generated_r in zip(input_rows, generations, strict=False)
        ]

        if benchmark_config.scoring_params is not None:
            scoring_functions_dict = {
                scoring_fn_id: benchmark_config.scoring_params.get(scoring_fn_id, None)
                for scoring_fn_id in scoring_functions
            }
        else:
            scoring_functions_dict = dict.fromkeys(scoring_functions)

        score_response = await self.scoring_api.score(
            input_rows=score_input_rows, scoring_functions=scoring_functions_dict
        )

        return EvaluateResponse(generations=generations, scores=score_response.results)

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        if job_id in self.jobs:
            return Job(job_id=job_id, status=JobStatus.completed)

        raise ValueError(f"Job {job_id} not found")

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        job = await self.job_status(benchmark_id, job_id)
        status = job.status
        if not status or status != JobStatus.completed:
            raise ValueError(f"Job is not completed, Status: {status.value}")

        return self.jobs[job_id]
