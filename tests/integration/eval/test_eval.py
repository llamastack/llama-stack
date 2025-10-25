# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import uuid
from pathlib import Path

import pytest

from ..datasets.test_datasets import data_url_from_file

# How to run this test:
#
# LLAMA_STACK_CONFIG="template-name" pytest -v tests/integration/eval


@pytest.mark.parametrize("scoring_fn_id", ["basic::equality"])
def test_evaluate_rows(llama_stack_client, text_model_id, scoring_fn_id):
    dataset = llama_stack_client.beta.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": data_url_from_file(Path(__file__).parent.parent / "datasets" / "test_dataset.csv"),
        },
    )
    response = llama_stack_client.beta.datasets.list()
    assert any(x.identifier == dataset.identifier for x in response)

    rows = llama_stack_client.beta.datasets.iterrows(
        dataset_id=dataset.identifier,
        limit=3,
    )
    assert len(rows.data) == 3

    scoring_functions = [
        scoring_fn_id,
    ]
    benchmark_id = str(uuid.uuid4())
    llama_stack_client.alpha.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset.identifier,
        scoring_functions=scoring_functions,
    )
    list_benchmarks = llama_stack_client.alpha.benchmarks.list()
    assert any(x.identifier == benchmark_id for x in list_benchmarks)

    response = llama_stack_client.alpha.eval.evaluate_rows(
        benchmark_id=benchmark_id,
        input_rows=rows.data,
        scoring_functions=scoring_functions,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": text_model_id,
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
    dataset = llama_stack_client.beta.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": data_url_from_file(Path(__file__).parent.parent / "datasets" / "test_dataset.csv"),
        },
    )
    benchmark_id = str(uuid.uuid4())
    llama_stack_client.alpha.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset.identifier,
        scoring_functions=[scoring_fn_id],
    )

    response = llama_stack_client.alpha.eval.run_eval(
        benchmark_id=benchmark_id,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": text_model_id,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            },
        },
    )
    assert response.job_id == "0"
    job_status = llama_stack_client.alpha.eval.jobs.status(job_id=response.job_id, benchmark_id=benchmark_id)
    assert job_status and job_status.status == "completed"

    eval_response = llama_stack_client.alpha.eval.jobs.retrieve(job_id=response.job_id, benchmark_id=benchmark_id)
    assert eval_response is not None
    assert len(eval_response.generations) == 5
    assert scoring_fn_id in eval_response.scores
