# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import MagicMock, patch

import pytest

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval.eval import BenchmarkConfig, EvaluateResponse, ModelCandidate, SamplingParams
from llama_stack.apis.inference.inference import TopPSamplingStrategy
from llama_stack.apis.resource import ResourceType
from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.remote.eval.nvidia.config import NVIDIAEvalConfig
from llama_stack.providers.remote.eval.nvidia.eval import NVIDIAEvalImpl

MOCK_DATASET_ID = "default/test-dataset"
MOCK_BENCHMARK_ID = "test-benchmark"


@pytest.fixture
def nvidia_eval_setup():
    """Set up the NVIDIA eval implementation with mocked dependencies."""
    os.environ["NVIDIA_EVALUATOR_URL"] = "http://nemo.test"

    # Create mock APIs
    datasetio_api = MagicMock()
    datasets_api = MagicMock()
    scoring_api = MagicMock()
    inference_api = MagicMock()
    agents_api = MagicMock()

    config = NVIDIAEvalConfig(
        evaluator_url=os.environ["NVIDIA_EVALUATOR_URL"],
    )

    eval_impl = NVIDIAEvalImpl(
        config=config,
        datasetio_api=datasetio_api,
        datasets_api=datasets_api,
        scoring_api=scoring_api,
        inference_api=inference_api,
        agents_api=agents_api,
    )

    # Mock the HTTP request methods
    with (
        patch("llama_stack.providers.remote.eval.nvidia.eval.NVIDIAEvalImpl._evaluator_get") as mock_evaluator_get,
        patch("llama_stack.providers.remote.eval.nvidia.eval.NVIDIAEvalImpl._evaluator_post") as mock_evaluator_post,
    ):
        yield {
            "eval_impl": eval_impl,
            "mock_evaluator_get": mock_evaluator_get,
            "mock_evaluator_post": mock_evaluator_post,
            "datasetio_api": datasetio_api,
            "datasets_api": datasets_api,
            "scoring_api": scoring_api,
            "inference_api": inference_api,
            "agents_api": agents_api,
        }


def _assert_request_body(mock_evaluator_post, expected_json):
    """Helper method to verify request body in Evaluator POST request is correct"""
    call_args = mock_evaluator_post.call_args
    actual_json = call_args[0][1]

    # Check that all expected keys contain the expected values in the actual JSON
    for key, value in expected_json.items():
        assert key in actual_json, f"Key '{key}' missing in actual JSON"

        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                assert nested_key in actual_json[key], f"Nested key '{nested_key}' missing in actual JSON['{key}']"
                assert actual_json[key][nested_key] == nested_value, f"Value mismatch for '{key}.{nested_key}'"
        else:
            assert actual_json[key] == value, f"Value mismatch for '{key}'"


async def test_register_benchmark(nvidia_eval_setup):
    eval_impl = nvidia_eval_setup["eval_impl"]
    mock_evaluator_post = nvidia_eval_setup["mock_evaluator_post"]

    eval_config = {
        "type": "custom",
        "params": {"parallelism": 8},
        "tasks": {
            "qa": {
                "type": "completion",
                "params": {"template": {"prompt": "{{prompt}}", "max_tokens": 200}},
                "dataset": {"files_url": f"hf://datasets/{MOCK_DATASET_ID}/testing/testing.jsonl"},
                "metrics": {"bleu": {"type": "bleu", "params": {"references": ["{{ideal_response}}"]}}},
            }
        },
    }

    benchmark = Benchmark(
        provider_id="nvidia",
        type=ResourceType.benchmark,
        identifier=MOCK_BENCHMARK_ID,
        dataset_id=MOCK_DATASET_ID,
        scoring_functions=["basic::equality"],
        metadata=eval_config,
    )

    # Mock Evaluator API response
    mock_evaluator_response = {"id": MOCK_BENCHMARK_ID, "status": "created"}
    mock_evaluator_post.return_value = mock_evaluator_response

    # Register the benchmark
    await eval_impl.register_benchmark(benchmark)

    # Verify the Evaluator API was called correctly
    mock_evaluator_post.assert_called_once()
    _assert_request_body(
        mock_evaluator_post, {"namespace": benchmark.provider_id, "name": benchmark.identifier, **eval_config}
    )


async def test_run_eval(nvidia_eval_setup):
    eval_impl = nvidia_eval_setup["eval_impl"]
    mock_evaluator_post = nvidia_eval_setup["mock_evaluator_post"]

    benchmark_config = BenchmarkConfig(
        eval_candidate=ModelCandidate(
            type="model",
            model=CoreModelId.llama3_1_8b_instruct.value,
            sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
        )
    )

    # Mock Evaluator API response
    mock_evaluator_response = {"id": "job-123", "status": "created"}
    mock_evaluator_post.return_value = mock_evaluator_response

    # Run the Evaluation job
    result = await eval_impl.run_eval(benchmark_id=MOCK_BENCHMARK_ID, benchmark_config=benchmark_config)

    # Verify the Evaluator API was called correctly
    mock_evaluator_post.assert_called_once()
    _assert_request_body(
        mock_evaluator_post,
        {
            "config": f"nvidia/{MOCK_BENCHMARK_ID}",
            "target": {"type": "model", "model": "Llama3.1-8B-Instruct"},
        },
    )

    # Verify the result
    assert isinstance(result, Job)
    assert result.job_id == "job-123"
    assert result.status == JobStatus.in_progress


async def test_job_status(nvidia_eval_setup):
    eval_impl = nvidia_eval_setup["eval_impl"]
    mock_evaluator_get = nvidia_eval_setup["mock_evaluator_get"]

    # Mock Evaluator API response
    mock_evaluator_response = {"id": "job-123", "status": "completed"}
    mock_evaluator_get.return_value = mock_evaluator_response

    # Get the Evaluation job
    result = await eval_impl.job_status(benchmark_id=MOCK_BENCHMARK_ID, job_id="job-123")

    # Verify the result
    assert isinstance(result, Job)
    assert result.job_id == "job-123"
    assert result.status == JobStatus.completed

    # Verify the API was called correctly
    mock_evaluator_get.assert_called_once_with(f"/v1/evaluation/jobs/{result.job_id}")


async def test_job_cancel(nvidia_eval_setup):
    eval_impl = nvidia_eval_setup["eval_impl"]
    mock_evaluator_post = nvidia_eval_setup["mock_evaluator_post"]

    # Mock Evaluator API response
    mock_evaluator_response = {"id": "job-123", "status": "cancelled"}
    mock_evaluator_post.return_value = mock_evaluator_response

    # Cancel the Evaluation job
    await eval_impl.job_cancel(benchmark_id=MOCK_BENCHMARK_ID, job_id="job-123")

    # Verify the API was called correctly
    mock_evaluator_post.assert_called_once_with("/v1/evaluation/jobs/job-123/cancel", {})


async def test_job_result(nvidia_eval_setup):
    eval_impl = nvidia_eval_setup["eval_impl"]
    mock_evaluator_get = nvidia_eval_setup["mock_evaluator_get"]

    # Mock Evaluator API responses
    mock_job_status_response = {"id": "job-123", "status": "completed"}
    mock_job_results_response = {
        "id": "job-123",
        "status": "completed",
        "results": {MOCK_BENCHMARK_ID: {"score": 0.85, "details": {"accuracy": 0.85, "f1": 0.84}}},
    }
    mock_evaluator_get.side_effect = [
        mock_job_status_response,  # First call to retrieve job
        mock_job_results_response,  # Second call to retrieve job results
    ]

    # Get the Evaluation job results
    result = await eval_impl.job_result(benchmark_id=MOCK_BENCHMARK_ID, job_id="job-123")

    # Verify the result
    assert isinstance(result, EvaluateResponse)
    assert MOCK_BENCHMARK_ID in result.scores
    assert result.scores[MOCK_BENCHMARK_ID].aggregated_results["results"][MOCK_BENCHMARK_ID]["score"] == 0.85

    # Verify the API was called correctly
    assert mock_evaluator_get.call_count == 2
    mock_evaluator_get.assert_any_call("/v1/evaluation/jobs/job-123")
    mock_evaluator_get.assert_any_call("/v1/evaluation/jobs/job-123/results")
