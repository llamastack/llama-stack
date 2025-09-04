# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import inspect
import itertools
import os
import platform
import textwrap
import time

import pytest
from dotenv import load_dotenv

from llama_stack.log import get_logger

logger = get_logger(__name__, category="tests")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.execution_outcome = report.outcome
        item.was_xfail = getattr(report, "wasxfail", False)


def pytest_sessionstart(session):
    # stop macOS from complaining about duplicate OpenMP libraries
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if "LLAMA_STACK_TEST_INFERENCE_MODE" not in os.environ:
        os.environ["LLAMA_STACK_TEST_INFERENCE_MODE"] = "replay"


def pytest_runtest_teardown(item):
    # Check if the test actually ran and passed or failed, but was not skipped or an expected failure (xfail)
    outcome = getattr(item, "execution_outcome", None)
    was_xfail = getattr(item, "was_xfail", False)

    name = item.nodeid
    if not any(x in name for x in ("inference/", "safety/", "agents/")):
        return

    logger.debug(f"Test '{item.nodeid}' outcome was '{outcome}' (xfail={was_xfail})")
    if outcome in ("passed", "failed") and not was_xfail:
        interval_seconds = os.getenv("LLAMA_STACK_TEST_INTERVAL_SECONDS")
        if interval_seconds:
            time.sleep(float(interval_seconds))


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True

    load_dotenv()

    env_vars = config.getoption("--env") or []
    for env_var in env_vars:
        key, value = env_var.split("=", 1)
        os.environ[key] = value

    if platform.system() == "Darwin":  # Darwin is the system name for macOS
        os.environ["DISABLE_CODE_SANDBOX"] = "1"
        logger.info("Setting DISABLE_CODE_SANDBOX=1 for macOS")


def pytest_addoption(parser):
    parser.addoption(
        "--stack-config",
        help=textwrap.dedent(
            """
            a 'pointer' to the stack. this can be either be:
            (a) a template name like `starter`, or
            (b) a path to a run.yaml file, or
            (c) an adhoc config spec, e.g. `inference=fireworks,safety=llama-guard,agents=meta-reference`
            """
        ),
    )
    parser.addoption("--env", action="append", help="Set environment variables, e.g. --env KEY=value")
    parser.addoption(
        "--text-model",
        help="comma-separated list of text models. Fixture name: text_model_id",
    )
    parser.addoption(
        "--vision-model",
        help="comma-separated list of vision models. Fixture name: vision_model_id",
    )
    parser.addoption(
        "--embedding-model",
        help="comma-separated list of embedding models. Fixture name: embedding_model_id",
    )
    parser.addoption(
        "--safety-shield",
        help="comma-separated list of safety shields. Fixture name: shield_id",
    )
    parser.addoption(
        "--judge-model",
        help="Specify the judge model to use for testing",
    )
    parser.addoption(
        "--embedding-dimension",
        type=int,
        default=384,
        help="Output dimensionality of the embedding model to use for testing. Default: 384",
    )
    parser.addoption(
        "--record-responses",
        action="store_true",
        help="Record new API responses instead of using cached ones.",
    )
    parser.addoption(
        "--report",
        help="Path where the test report should be written, e.g. --report=/path/to/report.md",
    )


MODEL_SHORT_IDS = {
    "meta-llama/Llama-3.2-3B-Instruct": "3B",
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
    "meta-llama/Llama-3.1-70B-Instruct": "70B",
    "meta-llama/Llama-3.1-405B-Instruct": "405B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "11B",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "90B",
    "meta-llama/Llama-3.3-70B-Instruct": "70B",
    "meta-llama/Llama-Guard-3-1B": "Guard1B",
    "meta-llama/Llama-Guard-3-8B": "Guard8B",
    "all-MiniLM-L6-v2": "MiniLM",
}


def get_short_id(value):
    return MODEL_SHORT_IDS.get(value, value)


def pytest_generate_tests(metafunc):
    """
    This is the main function which processes CLI arguments and generates various combinations of parameters.
    It is also responsible for generating test IDs which are succinct enough.

    Each option can be comma separated list of values which results in multiple parameter combinations.
    """
    params = []
    param_values = {}
    id_parts = []

    # Map of fixture name to its CLI option and ID prefix
    fixture_configs = {
        "text_model_id": ("--text-model", "txt"),
        "vision_model_id": ("--vision-model", "vis"),
        "embedding_model_id": ("--embedding-model", "emb"),
        "shield_id": ("--safety-shield", "shield"),
        "judge_model_id": ("--judge-model", "judge"),
        "embedding_dimension": ("--embedding-dimension", "dim"),
    }

    # Collect all parameters and their values
    for fixture_name, (option, id_prefix) in fixture_configs.items():
        if fixture_name not in metafunc.fixturenames:
            continue

        params.append(fixture_name)
        val = metafunc.config.getoption(option)

        values = [v.strip() for v in str(val).split(",")] if val else [None]
        param_values[fixture_name] = values
        if val:
            id_parts.extend(f"{id_prefix}={get_short_id(v)}" for v in values)

    if not params:
        return

    # Generate all combinations of parameter values
    value_combinations = list(itertools.product(*[param_values[p] for p in params]))

    # Generate test IDs
    test_ids = []
    non_empty_params = [(i, values) for i, values in enumerate(param_values.values()) if values[0] is not None]

    # Get actual function parameters using inspect
    test_func_params = set(inspect.signature(metafunc.function).parameters.keys())

    if non_empty_params:
        # For each combination, build an ID from the non-None parameters
        for combo in value_combinations:
            parts = []
            for param_name, val in zip(params, combo, strict=True):
                # Only include if parameter is in test function signature and value is meaningful
                if param_name in test_func_params and val:
                    prefix = fixture_configs[param_name][1]  # Get the ID prefix
                    parts.append(f"{prefix}={get_short_id(val)}")
            if parts:
                test_ids.append(":".join(parts))

    metafunc.parametrize(params, value_combinations, scope="session", ids=test_ids if test_ids else None)


pytest_plugins = ["tests.integration.fixtures.common"]
