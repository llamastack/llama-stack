# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Eval API integration tests.

These tests verify the Eval API integration by registering datasets and benchmarks,
then running evaluations. The tests prioritize using a model with existing recordings
(like `ollama/llama3.2:3b-instruct-fp16`) for replay compatibility and handle dynamic
model registration if the model is not already available.
"""

import time
import uuid
from pathlib import Path

import pytest
from llama_stack_client import LlamaStackClient

from ..datasets.test_datasets import data_url_from_file


def ensure_model_registered(client: LlamaStackClient, model_id: str) -> str:
    """Ensure the specified model is registered, registering it if necessary.

    Args:
        client: The llama stack client instance.
        model_id: The model ID to ensure is registered.

    Returns:
        The model ID that was ensured/registered.
    """
    models = client.models.list()
    if any(m.identifier == model_id for m in models):
        return model_id

    providers = client.providers.list()
    inference_providers = [p for p in providers if p.api == "inference"]
    for p in inference_providers:
        try:
            client.models.register(model_id=model_id, provider_id=p.provider_id)
            return model_id
        except Exception:
            continue
    return model_id


@pytest.mark.parametrize("scoring_fn_id", ["basic::equality"])
def test_evaluate_rows(llama_stack_client, text_model_id, scoring_fn_id):
    """Test the evaluate_rows endpoint (POST /eval/benchmarks/{benchmark_id}/evaluations)."""
    dataset = llama_stack_client.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": data_url_from_file(str(Path(__file__).parent.parent / "datasets" / "test_dataset.csv")),
        },
    )
    response = llama_stack_client.datasets.list()
    assert any(x.identifier == dataset.identifier for x in response)

    scoring_functions = [scoring_fn_id]
    benchmark_id = str(uuid.uuid4())
    llama_stack_client.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset.identifier,
        scoring_functions=scoring_functions,
    )
    list_benchmarks = llama_stack_client.benchmarks.list()
    assert any(x.identifier == benchmark_id for x in list_benchmarks)

    if hasattr(llama_stack_client.datasets, "iterrows"):
        rows_response = llama_stack_client.datasets.iterrows(dataset_id=dataset.identifier, limit=3)
        input_rows = rows_response.data
    else:
        pytest.fail("datasets.iterrows not found on client")

    actual_model_id = text_model_id or "ollama/llama3.2:3b-instruct-fp16"
    ensure_model_registered(llama_stack_client, actual_model_id)

    response = llama_stack_client.alpha.eval.evaluate_rows(
        benchmark_id=benchmark_id,
        input_rows=input_rows,
        scoring_functions=scoring_functions,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": actual_model_id,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            },
        },
    )

    assert len(response.generations) == 3
    assert scoring_fn_id in response.scores


@pytest.mark.parametrize("scoring_fn_id", ["basic::subset_of"])
def test_evaluate_benchmark(llama_stack_client, text_model_id, scoring_fn_id):
    """Test run_eval, job_status, and job_result endpoints.

    Verifies:
    - POST /eval/benchmarks/{benchmark_id}/jobs (run_eval)
    - GET /eval/benchmarks/{benchmark_id}/jobs/{job_id} (job_status)
    - GET /eval/benchmarks/{benchmark_id}/jobs/{job_id}/result (job_result)
    """
    dataset = llama_stack_client.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": data_url_from_file(str(Path(__file__).parent.parent / "datasets" / "test_dataset.csv")),
        },
    )
    benchmark_id = str(uuid.uuid4())
    llama_stack_client.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset.identifier,
        scoring_functions=[scoring_fn_id],
    )

    actual_model_id = text_model_id or "ollama/llama3.2:3b-instruct-fp16"
    ensure_model_registered(llama_stack_client, actual_model_id)

    # Start the evaluation job
    response = llama_stack_client.alpha.eval.run_eval(
        benchmark_id=benchmark_id,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": actual_model_id,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            },
        },
    )
    assert response.job_id is not None
    job_id = response.job_id

    # Poll for job completion (GET /eval/benchmarks/{benchmark_id}/jobs/{job_id})
    max_wait_seconds = 60
    poll_interval = 1
    elapsed = 0
    job_status = None
    while elapsed < max_wait_seconds:
        job_status = llama_stack_client.alpha.eval.jobs.status(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )
        if job_status.status in ("completed", "failed"):
            break
        time.sleep(poll_interval)
        elapsed += poll_interval

    assert job_status is not None
    assert job_status.status == "completed", f"Failed to complete job in time, status: {job_status.status}"

    # Retrieve the job result (GET /eval/benchmarks/{benchmark_id}/jobs/{job_id}/result)
    result = llama_stack_client.alpha.eval.jobs.retrieve(
        benchmark_id=benchmark_id,
        job_id=job_id,
    )
    assert result is not None
    assert hasattr(result, "generations") or hasattr(result, "scores")


@pytest.mark.parametrize("scoring_fn_id", ["basic::equality"])
def test_cancel_eval(llama_stack_client, text_model_id, scoring_fn_id):
    """Test the job cancel endpoint (DELETE /eval/benchmarks/{benchmark_id}/jobs/{job_id}).

    This test verifies that the cancel endpoint is correctly wired up and reachable.
    Currently, job cancellation is not implemented in the eval provider, so we expect
    a NotImplementedError to be raised. This test will be updated once the feature
    is fully implemented.
    """
    dataset = llama_stack_client.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": data_url_from_file(str(Path(__file__).parent.parent / "datasets" / "test_dataset.csv")),
        },
    )
    benchmark_id = str(uuid.uuid4())
    llama_stack_client.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset.identifier,
        scoring_functions=[scoring_fn_id],
    )

    actual_model_id = text_model_id or "ollama/llama3.2:3b-instruct-fp16"
    ensure_model_registered(llama_stack_client, actual_model_id)

    # Start the evaluation job
    response = llama_stack_client.alpha.eval.run_eval(
        benchmark_id=benchmark_id,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": actual_model_id,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            },
        },
    )
    assert response.job_id is not None
    job_id = response.job_id

    # Try to cancel the job (DELETE /eval/benchmarks/{benchmark_id}/jobs/{job_id})
    # Currently, job cancellation is not implemented, so we expect NotImplementedError.
    # This verifies the endpoint is wired up correctly even though the feature isn't complete.
    with pytest.raises(Exception) as exc_info:
        llama_stack_client.alpha.eval.jobs.cancel(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )
    # The error message should indicate the feature is not implemented
    assert "not implemented" in str(exc_info.value).lower()

    # Verify the job still has a valid status (wasn't corrupted by the cancel attempt)
    job_status = llama_stack_client.alpha.eval.jobs.status(
        benchmark_id=benchmark_id,
        job_id=job_id,
    )
    # The status should be 'completed' since jobs run synchronously in CI
    assert job_status.status in ("completed", "failed", "running")
