# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from llama_stack_api.eval.models import (
    BenchmarkConfig,
    EvaluateResponse,
    EvaluateRowsRequest,
    ModelCandidate,
    RunEvalRequest,
)
from llama_stack_api.inference import SamplingParams, TopPSamplingStrategy
from llama_stack_api.scoring import ScoringResult


def test_model_candidate_valid():
    mc = ModelCandidate(
        model="test-model",
        sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
    )
    assert mc.model == "test-model"
    assert mc.type == "model"


def test_benchmark_config_valid():
    mc = ModelCandidate(
        model="test-model",
        sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
    )
    bc = BenchmarkConfig(eval_candidate=mc, num_examples=5)
    assert bc.num_examples == 5
    assert bc.scoring_params == {}


def test_evaluate_response_valid():
    er = EvaluateResponse(
        generations=[{"input": "test", "output": "result"}],
        scores={
            "accuracy": ScoringResult(
                score_rows=[{"score": 0.9}],
                aggregated_results={"average": 0.9},
            )
        },
    )
    assert len(er.generations) == 1
    assert "accuracy" in er.scores


def test_run_eval_request_valid():
    mc = ModelCandidate(
        model="test-model",
        sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
    )
    bc = BenchmarkConfig(eval_candidate=mc)
    req = RunEvalRequest(benchmark_id="bench-123", benchmark_config=bc)
    assert req.benchmark_id == "bench-123"


def test_evaluate_rows_request_empty_arrays_fail():
    mc = ModelCandidate(
        model="test-model",
        sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
    )
    bc = BenchmarkConfig(eval_candidate=mc)

    with pytest.raises(ValidationError):
        EvaluateRowsRequest(
            benchmark_id="bench-123",
            input_rows=[],
            scoring_functions=["func1"],
            benchmark_config=bc,
        )

    with pytest.raises(ValidationError):
        EvaluateRowsRequest(
            benchmark_id="bench-123",
            input_rows=[{"test": "data"}],
            scoring_functions=[],
            benchmark_config=bc,
        )
