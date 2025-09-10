# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from typing import Any

from tqdm import tqdm

from llama_stack.apis.agents import Agents, StepType
from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference, SystemMessage, UserMessage
from llama_stack.apis.scoring import Scoring
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.providers.inline.agents.meta_reference.agent_instance import (
    MEMORY_QUERY_TOOL,
)
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

    async def _run_agent_generation(
        self, input_rows: list[dict[str, Any]], benchmark_config: BenchmarkConfig
    ) -> list[dict[str, Any]]:
        candidate = benchmark_config.eval_candidate
        create_response = await self.agents_api.create_agent(candidate.config)
        agent_id = create_response.agent_id

        generations = []
        for i, x in tqdm(enumerate(input_rows)):
            assert ColumnName.chat_completion_input.value in x, "Invalid input row"
            input_messages = json.loads(x[ColumnName.chat_completion_input.value])
            input_messages = [UserMessage(**x) for x in input_messages if x["role"] == "user"]

            # NOTE: only single-turn agent generation is supported. Create a new session for each input row
            session_create_response = await self.agents_api.create_agent_session(agent_id, f"session-{i}")
            session_id = session_create_response.session_id

            turn_request = dict(
                agent_id=agent_id,
                session_id=session_id,
                messages=input_messages,
                stream=True,
            )
            turn_response = [chunk async for chunk in await self.agents_api.create_agent_turn(**turn_request)]
            final_event = turn_response[-1].event.payload

            # check if there's a memory retrieval step and extract the context
            memory_rag_context = None
            for step in final_event.turn.steps:
                if step.step_type == StepType.tool_execution.value:
                    for tool_response in step.tool_responses:
                        if tool_response.tool_name == MEMORY_QUERY_TOOL:
                            memory_rag_context = " ".join(x.text for x in tool_response.content)

            agent_generation = {}
            agent_generation[ColumnName.generated_answer.value] = final_event.turn.output_message.content
            if memory_rag_context:
                agent_generation[ColumnName.context.value] = memory_rag_context

            generations.append(agent_generation)

        return generations

    async def _run_model_generation(
        self, input_rows: list[dict[str, Any]], benchmark_config: BenchmarkConfig
    ) -> list[dict[str, Any]]:
        candidate = benchmark_config.eval_candidate
        assert candidate.sampling_params.max_tokens is not None, "SamplingParams.max_tokens must be provided"

        generations = []
        for x in tqdm(input_rows):
            if ColumnName.completion_input.value in x:
                input_content = json.loads(x[ColumnName.completion_input.value])
                response = await self.inference_api.completion(
                    model=candidate.model,
                    content=input_content,
                    sampling_params=candidate.sampling_params,
                )
                generations.append({ColumnName.generated_answer.value: response.completion_message.content})
            elif ColumnName.chat_completion_input.value in x:
                chat_completion_input_json = json.loads(x[ColumnName.chat_completion_input.value])
                input_messages = [UserMessage(**x) for x in chat_completion_input_json if x["role"] == "user"]
                messages = []
                if candidate.system_message:
                    messages.append(candidate.system_message)
                messages += [SystemMessage(**x) for x in chat_completion_input_json if x["role"] == "system"]
                messages += input_messages
                response = await self.inference_api.chat_completion(
                    model_id=candidate.model,
                    messages=messages,
                    sampling_params=candidate.sampling_params,
                )
                generations.append({ColumnName.generated_answer.value: response.completion_message.content})
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
        if candidate.type == "agent":
            generations = await self._run_agent_generation(input_rows, benchmark_config)
        elif candidate.type == "model":
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
