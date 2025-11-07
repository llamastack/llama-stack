# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from llama_stack.apis.common.job_types import Job
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.apis.scoring import ScoringResult
from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.apis.version import LLAMA_STACK_API_V1ALPHA
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ModelCandidate(BaseModel):
    """A model candidate for evaluation.

    :param model: The model ID to evaluate.
    :param sampling_params: The sampling parameters for the model.
    :param system_message: (Optional) The system message providing instructions or context to the model.
    """

    type: Literal["model"] = "model"
    model: str
    sampling_params: SamplingParams
    system_message: SystemMessage | None = None


EvalCandidate = ModelCandidate


@json_schema_type
class BenchmarkConfig(BaseModel):
    """A benchmark configuration for evaluation.

    :param eval_candidate: The candidate to evaluate.
    :param scoring_params: Map between scoring function id and parameters for each scoring function you want to run
    :param num_examples: (Optional) The number of examples to evaluate. If not provided, all examples in the dataset will be evaluated
    """

    eval_candidate: EvalCandidate
    scoring_params: dict[str, ScoringFnParams] = Field(
        description="Map between scoring function id and parameters for each scoring function you want to run",
        default_factory=dict,
    )
    num_examples: int | None = Field(
        description="Number of examples to evaluate (useful for testing), if not provided, all examples in the dataset will be evaluated",
        default=None,
    )
    # we could optinally add any specific dataset config here


@json_schema_type
class EvaluateResponse(BaseModel):
    """The response from an evaluation.

    :param generations: The generations from the evaluation.
    :param scores: The scores from the evaluation.
    """

    generations: list[dict[str, Any]]
    # each key in the dict is a scoring function name
    scores: dict[str, ScoringResult]


class Eval(Protocol):
    """Evaluations

    Llama Stack Evaluation API for running evaluations on model and agent candidates."""

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        """Run an evaluation on a benchmark.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param benchmark_config: The configuration for the benchmark.
        :returns: The job that was created to run the evaluation.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/evaluations", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param input_rows: The rows to evaluate.
        :param scoring_functions: The scoring functions to use for the evaluation.
        :param benchmark_config: The configuration for the benchmark.
        :returns: EvaluateResponse object containing generations and scores.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        """Get the status of a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to get the status of.
        :returns: The status of the evaluation job.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}", method="DELETE", level=LLAMA_STACK_API_V1ALPHA)
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to cancel.
        """
        ...

    @webmethod(
        route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}/result", method="GET", level=LLAMA_STACK_API_V1ALPHA
    )
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to get the result of.
        :returns: The result of the job.
        """
        ...
