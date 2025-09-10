# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.core.stack import replace_env_vars


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
