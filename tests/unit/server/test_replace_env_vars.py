# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.core.stack import EnvVarError, replace_env_vars


@pytest.fixture
def setup_env_vars():
    # Clear any existing environment variables we'll use in tests
    for var in ["TEST_VAR", "EMPTY_VAR", "ZERO_VAR"]:
        if var in os.environ:
            del os.environ[var]

    # Set up test environment variables
    os.environ["TEST_VAR"] = "test_value"
    os.environ["EMPTY_VAR"] = ""
    os.environ["ZERO_VAR"] = "0"

    yield

    # Cleanup after test
    for var in ["TEST_VAR", "EMPTY_VAR", "ZERO_VAR"]:
        if var in os.environ:
            del os.environ[var]


def test_simple_replacement(setup_env_vars):
    assert replace_env_vars("${env.TEST_VAR}") == "test_value"


def test_simple_replacement_raises_when_not_set(setup_env_vars):
    """Test that ${env.VAR} without operators raises EnvVarError when env var is not set."""
    with pytest.raises(EnvVarError) as exc_info:
        replace_env_vars("${env.NOT_SET}")
    assert exc_info.value.var_name == "NOT_SET"


def test_default_value_when_not_set(setup_env_vars):
    assert replace_env_vars("${env.NOT_SET:=default}") == "default"


def test_default_value_when_set(setup_env_vars):
    assert replace_env_vars("${env.TEST_VAR:=default}") == "test_value"


def test_default_value_when_empty(setup_env_vars):
    assert replace_env_vars("${env.EMPTY_VAR:=default}") == "default"


def test_none_value_when_empty(setup_env_vars):
    assert replace_env_vars("${env.EMPTY_VAR:=}") is None


def test_value_when_set(setup_env_vars):
    assert replace_env_vars("${env.TEST_VAR:=}") == "test_value"


def test_empty_var_no_default(setup_env_vars):
    assert replace_env_vars("${env.EMPTY_VAR_NO_DEFAULT:+}") is None


def test_conditional_value_when_set(setup_env_vars):
    assert replace_env_vars("${env.TEST_VAR:+conditional}") == "conditional"


def test_conditional_value_when_not_set(setup_env_vars):
    assert replace_env_vars("${env.NOT_SET:+conditional}") is None


def test_conditional_value_when_empty(setup_env_vars):
    assert replace_env_vars("${env.EMPTY_VAR:+conditional}") is None


def test_conditional_value_with_zero(setup_env_vars):
    assert replace_env_vars("${env.ZERO_VAR:+conditional}") == "conditional"


def test_mixed_syntax(setup_env_vars):
    assert replace_env_vars("${env.TEST_VAR:=default} and ${env.NOT_SET:+conditional}") == "test_value and "
    assert replace_env_vars("${env.NOT_SET:=default} and ${env.TEST_VAR:+conditional}") == "default and conditional"


def test_nested_structures(setup_env_vars):
    data = {
        "key1": "${env.TEST_VAR:=default}",
        "key2": ["${env.NOT_SET:=default}", "${env.TEST_VAR:+conditional}"],
        "key3": {"nested": "${env.NOT_SET:+conditional}"},
    }
    expected = {"key1": "test_value", "key2": ["default", "conditional"], "key3": {"nested": None}}
    assert replace_env_vars(data) == expected


def test_explicit_strings_preserved(setup_env_vars):
    # Explicit strings that look like numbers/booleans should remain strings
    data = {"port": "8080", "enabled": "true", "count": "123", "ratio": "3.14"}
    expected = {"port": "8080", "enabled": "true", "count": "123", "ratio": "3.14"}
    assert replace_env_vars(data) == expected


def test_resource_with_empty_benchmark_id_skipped(setup_env_vars):
    """Test that resources with empty benchmark_id from conditional env vars are skipped."""
    data = {
        "benchmarks": [
            {"benchmark_id": "${env.BENCHMARK_ID:+my-benchmark}", "dataset_id": "test-dataset"},
            {"benchmark_id": "always-present", "dataset_id": "another-dataset"},
        ]
    }
    # BENCHMARK_ID is not set, so first benchmark should be skipped
    result = replace_env_vars(data)
    assert len(result["benchmarks"]) == 1
    assert result["benchmarks"][0]["benchmark_id"] == "always-present"


def test_resource_with_set_benchmark_id_not_skipped(setup_env_vars):
    """Test that resources with set benchmark_id are not skipped."""
    os.environ["BENCHMARK_ID"] = "enabled"
    try:
        data = {
            "benchmarks": [
                {"benchmark_id": "${env.BENCHMARK_ID:+my-benchmark}", "dataset_id": "test-dataset"},
                {"benchmark_id": "always-present", "dataset_id": "another-dataset"},
            ]
        }
        result = replace_env_vars(data)
        assert len(result["benchmarks"]) == 2
        assert result["benchmarks"][0]["benchmark_id"] == "my-benchmark"
        assert result["benchmarks"][1]["benchmark_id"] == "always-present"
    finally:
        del os.environ["BENCHMARK_ID"]


def test_resource_with_empty_model_id_skipped(setup_env_vars):
    """Test that resources with empty model_id from conditional env vars are skipped."""
    data = {
        "models": [
            {"model_id": "${env.MODEL_ID:+my-model}", "provider_id": "test-provider"},
            {"model_id": "always-present", "provider_id": "another-provider"},
        ]
    }
    # MODEL_ID is not set, so first model should be skipped
    result = replace_env_vars(data)
    assert len(result["models"]) == 1
    assert result["models"][0]["model_id"] == "always-present"


def test_resource_with_empty_shield_id_skipped(setup_env_vars):
    """Test that resources with empty shield_id from conditional env vars are skipped."""
    data = {
        "shields": [
            {"shield_id": "${env.SHIELD_ID:+my-shield}", "provider_id": "test-provider"},
            {"shield_id": "always-present", "provider_id": "another-provider"},
        ]
    }
    # SHIELD_ID is not set, so first shield should be skipped
    result = replace_env_vars(data)
    assert len(result["shields"]) == 1
    assert result["shields"][0]["shield_id"] == "always-present"


def test_multiple_resources_with_conditional_ids(setup_env_vars):
    """Test that multiple resource types with conditional IDs are handled correctly."""
    os.environ["INCLUDE_BENCHMARK"] = "yes"
    try:
        data = {
            "benchmarks": [
                {"benchmark_id": "${env.INCLUDE_BENCHMARK:+included-benchmark}", "dataset_id": "ds1"},
                {"benchmark_id": "${env.EXCLUDE_BENCHMARK:+excluded-benchmark}", "dataset_id": "ds2"},
            ],
            "models": [
                {"model_id": "${env.EXCLUDE_MODEL:+excluded-model}", "provider_id": "p1"},
            ],
        }
        result = replace_env_vars(data)
        # Only the benchmark with INCLUDE_BENCHMARK set should remain
        assert len(result["benchmarks"]) == 1
        assert result["benchmarks"][0]["benchmark_id"] == "included-benchmark"
        # Model with unset env var should be skipped
        assert len(result["models"]) == 0
    finally:
        del os.environ["INCLUDE_BENCHMARK"]
