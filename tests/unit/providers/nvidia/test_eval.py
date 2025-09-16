# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval.eval import BenchmarkConfig, EvaluateResponse, ModelCandidate, SamplingParams
from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.remote.eval.nvidia.config import NVIDIAEvalConfig
from llama_stack.providers.remote.eval.nvidia.eval import NVIDIAEvalImpl

MOCK_DATASET_ID = "default/test-dataset"
MOCK_BENCHMARK_ID = "test-benchmark"


class TestNVIDIAEvalImpl(unittest.TestCase):
    def setUp(self):
        os.environ["NVIDIA_EVALUATOR_URL"] = "http://nemo.test"

        # Create mock APIs
        self.datasetio_api = MagicMock()
        self.datasets_api = MagicMock()
        self.scoring_api = MagicMock()
        self.inference_api = MagicMock()
        self.agents_api = MagicMock()

        self.config = NVIDIAEvalConfig(
            evaluator_url=os.environ["NVIDIA_EVALUATOR_URL"],
        )

        self.eval_impl = NVIDIAEvalImpl(
            config=self.config,
            datasetio_api=self.datasetio_api,
            datasets_api=self.datasets_api,
            scoring_api=self.scoring_api,
            inference_api=self.inference_api,
            agents_api=self.agents_api,
        )

        # Mock the HTTP request methods
        self.evaluator_get_patcher = patch(
            "llama_stack.providers.remote.eval.nvidia.eval.NVIDIAEvalImpl._evaluator_get"
        )
        self.evaluator_post_patcher = patch(
            "llama_stack.providers.remote.eval.nvidia.eval.NVIDIAEvalImpl._evaluator_post"
        )
        self.evaluator_delete_patcher = patch(
            "llama_stack.providers.remote.eval.nvidia.eval.NVIDIAEvalImpl._evaluator_delete"
        )

        self.mock_evaluator_get = self.evaluator_get_patcher.start()
        self.mock_evaluator_post = self.evaluator_post_patcher.start()
        self.mock_evaluator_delete = self.evaluator_delete_patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        self.evaluator_get_patcher.stop()
        self.evaluator_post_patcher.stop()
        self.evaluator_delete_patcher.stop()

    def _assert_request_body(self, expected_json):
        """Helper method to verify request body in Evaluator POST request is correct"""
        call_args = self.mock_evaluator_post.call_args
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

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, run_async):
        self.run_async = run_async

    def test_register_benchmark(self):
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
            type="benchmark",
            identifier=MOCK_BENCHMARK_ID,
            dataset_id=MOCK_DATASET_ID,
            scoring_functions=["basic::equality"],
            metadata=eval_config,
        )

        # Mock Evaluator API response
        mock_evaluator_response = {"id": MOCK_BENCHMARK_ID, "status": "created"}
        self.mock_evaluator_post.return_value = mock_evaluator_response

        # Register the benchmark
        self.run_async(self.eval_impl.register_benchmark(benchmark))

        # Verify the Evaluator API was called correctly
        self.mock_evaluator_post.assert_called_once()
        self._assert_request_body({"namespace": benchmark.provider_id, "name": benchmark.identifier, **eval_config})

    def test_unregister_benchmark(self):
        # Unregister the benchmark
        self.run_async(self.eval_impl.unregister_benchmark(benchmark_id=MOCK_BENCHMARK_ID))

        # Verify the Evaluator API was called correctly
        self.mock_evaluator_delete.assert_called_once_with(f"/v1/evaluation/configs/nvidia/{MOCK_BENCHMARK_ID}")

    def test_run_eval(self):
        benchmark_config = BenchmarkConfig(
            eval_candidate=ModelCandidate(
                type="model",
                model=CoreModelId.llama3_1_8b_instruct.value,
                sampling_params=SamplingParams(max_tokens=100, temperature=0.7),
            )
        )

        # Mock Evaluator API response
        mock_evaluator_response = {"id": "job-123", "status": "created"}
        self.mock_evaluator_post.return_value = mock_evaluator_response

        # Run the Evaluation job
        result = self.run_async(
            self.eval_impl.run_eval(benchmark_id=MOCK_BENCHMARK_ID, benchmark_config=benchmark_config)
        )

        # Verify the Evaluator API was called correctly
        self.mock_evaluator_post.assert_called_once()
        self._assert_request_body(
            {
                "config": f"nvidia/{MOCK_BENCHMARK_ID}",
                "target": {"type": "model", "model": "meta/llama-3.1-8b-instruct"},
            }
        )

        # Verify the result
        assert isinstance(result, Job)
        assert result.job_id == "job-123"
        assert result.status == JobStatus.in_progress

    def test_job_status(self):
        # Mock Evaluator API response
        mock_evaluator_response = {"id": "job-123", "status": "completed"}
        self.mock_evaluator_get.return_value = mock_evaluator_response

        # Get the Evaluation job
        result = self.run_async(self.eval_impl.job_status(benchmark_id=MOCK_BENCHMARK_ID, job_id="job-123"))

        # Verify the result
        assert isinstance(result, Job)
        assert result.job_id == "job-123"
        assert result.status == JobStatus.completed

        # Verify the API was called correctly
        self.mock_evaluator_get.assert_called_once_with(f"/v1/evaluation/jobs/{result.job_id}")

    def test_job_cancel(self):
        # Mock Evaluator API response
        mock_evaluator_response = {"id": "job-123", "status": "cancelled"}
        self.mock_evaluator_post.return_value = mock_evaluator_response

        # Cancel the Evaluation job
        self.run_async(self.eval_impl.job_cancel(benchmark_id=MOCK_BENCHMARK_ID, job_id="job-123"))

        # Verify the API was called correctly
        self.mock_evaluator_post.assert_called_once_with("/v1/evaluation/jobs/job-123/cancel", {})

    def test_job_result(self):
        # Mock Evaluator API responses
        mock_job_status_response = {"id": "job-123", "status": "completed"}
        mock_job_results_response = {
            "id": "job-123",
            "status": "completed",
            "results": {MOCK_BENCHMARK_ID: {"score": 0.85, "details": {"accuracy": 0.85, "f1": 0.84}}},
        }
        self.mock_evaluator_get.side_effect = [
            mock_job_status_response,  # First call to retrieve job
            mock_job_results_response,  # Second call to retrieve job results
        ]

        # Get the Evaluation job results
        result = self.run_async(self.eval_impl.job_result(benchmark_id=MOCK_BENCHMARK_ID, job_id="job-123"))

        # Verify the result
        assert isinstance(result, EvaluateResponse)
        assert MOCK_BENCHMARK_ID in result.scores
        assert result.scores[MOCK_BENCHMARK_ID].aggregated_results["results"][MOCK_BENCHMARK_ID]["score"] == 0.85

        # Verify the API was called correctly
        assert self.mock_evaluator_get.call_count == 2
        self.mock_evaluator_get.assert_any_call("/v1/evaluation/jobs/job-123")
        self.mock_evaluator_get.assert_any_call("/v1/evaluation/jobs/job-123/results")
