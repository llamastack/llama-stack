# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from pathlib import Path

import pandas as pd
import pytest
import requests


@pytest.fixture
def sample_judge_prompt_template():
    return "Output a number response in the following format: Score: <number>, where <number> is the number between 0 and 9."


@pytest.fixture
def sample_scoring_fn_id():
    return "llm-as-judge-test-prompt"


def register_scoring_function(
    llama_stack_client,
    provider_id,
    scoring_fn_id,
    judge_model_id,
    judge_prompt_template,
):
    llama_stack_client.scoring_functions.register(
        scoring_fn_id=scoring_fn_id,
        provider_id=provider_id,
        description="LLM as judge scoring function with test prompt",
        return_type={
            "type": "string",
        },
        params={
            "type": "llm_as_judge",
            "judge_model": judge_model_id,
            "prompt_template": judge_prompt_template,
        },
    )


def test_scoring_functions_list(llama_stack_client):
    response = llama_stack_client.scoring_functions.list()
    assert isinstance(response, list)
    assert len(response) > 0


def test_scoring_functions_register(
    llama_stack_client,
    sample_scoring_fn_id,
    judge_model_id,
    sample_judge_prompt_template,
):
    llm_as_judge_provider = [
        x
        for x in llama_stack_client.providers.list()
        if x.api == "scoring" and x.provider_type == "inline::llm-as-judge"
    ]
    if len(llm_as_judge_provider) == 0:
        pytest.skip("No llm-as-judge provider found, cannot test registeration")

    llm_as_judge_provider_id = llm_as_judge_provider[0].provider_id
    register_scoring_function(
        llama_stack_client,
        llm_as_judge_provider_id,
        sample_scoring_fn_id,
        judge_model_id,
        sample_judge_prompt_template,
    )

    list_response = llama_stack_client.scoring_functions.list()
    assert isinstance(list_response, list)
    assert len(list_response) > 0
    assert any(x.identifier == sample_scoring_fn_id for x in list_response)


def test_scoring_functions_unregister(
    llama_stack_client,
    sample_scoring_fn_id,
    judge_model_id,
    sample_judge_prompt_template,
):
    llm_as_judge_provider = [
        x
        for x in llama_stack_client.providers.list()
        if x.api == "scoring" and x.provider_type == "inline::llm-as-judge"
    ]
    if len(llm_as_judge_provider) == 0:
        pytest.skip("No llm-as-judge provider found, cannot test unregister")

    llm_as_judge_provider_id = llm_as_judge_provider[0].provider_id

    # Register first
    register_scoring_function(
        llama_stack_client,
        llm_as_judge_provider_id,
        sample_scoring_fn_id,
        judge_model_id,
        sample_judge_prompt_template,
    )

    # Ensure it is present
    list_response = llama_stack_client.scoring_functions.list()
    assert any(x.identifier == sample_scoring_fn_id for x in list_response)

    # Unregister scoring fn
    try:
        base_url = llama_stack_client.base_url
    except AttributeError:
        pytest.skip("No server base_url available; cannot test HTTP unregister in library mode")

    resp = requests.delete(f"{base_url}/v1/scoring-functions/{sample_scoring_fn_id}", timeout=30)
    assert resp.status_code in (200, 204)
    list_after = llama_stack_client.scoring_functions.list()
    assert all(x.identifier != sample_scoring_fn_id for x in list_after)


@pytest.mark.parametrize("scoring_fn_id", ["basic::equality"])
def test_scoring_score(llama_stack_client, scoring_fn_id):
    # scoring individual rows
    df = pd.read_csv(Path(__file__).parent.parent / "datasets" / "test_dataset.csv")
    rows = df.to_dict(orient="records")

    scoring_functions = {
        scoring_fn_id: None,
    }

    response = llama_stack_client.scoring.score(
        input_rows=rows,
        scoring_functions=scoring_functions,
    )
    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == len(rows)


def test_scoring_score_with_params_llm_as_judge(
    llama_stack_client,
    sample_judge_prompt_template,
    judge_model_id,
):
    # scoring individual rows
    df = pd.read_csv(Path(__file__).parent.parent / "datasets" / "test_dataset.csv")
    rows = df.to_dict(orient="records")

    scoring_functions = {
        "llm-as-judge::base": dict(
            type="llm_as_judge",
            judge_model=judge_model_id,
            prompt_template=sample_judge_prompt_template,
            judge_score_regexes=[r"Score: (\d+)"],
            aggregation_functions=[
                "categorical_count",
            ],
        )
    }

    response = llama_stack_client.scoring.score(
        input_rows=rows,
        scoring_functions=scoring_functions,
    )
    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == len(rows)


@pytest.mark.parametrize(
    "provider_id",
    [
        "basic",
        "llm-as-judge",
        "braintrust",
    ],
)
def test_scoring_score_with_aggregation_functions(
    llama_stack_client,
    sample_judge_prompt_template,
    judge_model_id,
    provider_id,
    rag_dataset_for_test,
):
    df = pd.read_csv(Path(__file__).parent.parent / "datasets" / "test_dataset.csv")
    rows = df.to_dict(orient="records")

    scoring_fns_list = [x for x in llama_stack_client.scoring_functions.list() if x.provider_id == provider_id]
    if len(scoring_fns_list) == 0:
        pytest.skip(f"No scoring functions found for provider {provider_id}, skipping")

    scoring_functions = {}
    aggr_fns = [
        "accuracy",
        "median",
        "categorical_count",
        "average",
    ]

    scoring_fn = scoring_fns_list[0]
    if scoring_fn.provider_id == "llm-as-judge":
        aggr_fns = ["categorical_count"]
        scoring_functions[scoring_fn.identifier] = dict(
            type="llm_as_judge",
            judge_model=judge_model_id,
            prompt_template=sample_judge_prompt_template,
            judge_score_regexes=[r"Score: (\d+)"],
            aggregation_functions=aggr_fns,
        )
    elif scoring_fn.provider_id == "basic" or scoring_fn.provider_id == "braintrust":
        if "regex_parser" in scoring_fn.identifier:
            scoring_functions[scoring_fn.identifier] = dict(
                type="regex_parser",
                parsing_regexes=[r"Score: (\d+)"],
                aggregation_functions=aggr_fns,
            )
        else:
            scoring_functions[scoring_fn.identifier] = dict(
                type="basic",
                aggregation_functions=aggr_fns,
            )
    else:
        scoring_functions[scoring_fn.identifier] = None

    response = llama_stack_client.scoring.score(
        input_rows=rows,
        scoring_functions=scoring_functions,
    )

    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == len(rows)
        assert len(response.results[x].aggregated_results) == len(aggr_fns)
