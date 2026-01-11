# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import inspect
import itertools
import logging  # allow-direct-logging
import os
import tempfile
import textwrap
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from llama_stack.log import get_logger
from llama_stack.testing.api_recorder import patch_httpx_for_test_id

from .suites import SETUP_DEFINITIONS, SUITE_DEFINITIONS

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

    if "LLAMA_STACK_LOGGING" not in os.environ:
        os.environ["LLAMA_STACK_LOGGING"] = "all=warning"

    if "SQLITE_STORE_DIR" not in os.environ:
        os.environ["SQLITE_STORE_DIR"] = tempfile.mkdtemp()
        logger.info(f"Setting SQLITE_STORE_DIR: {os.environ['SQLITE_STORE_DIR']}")

    # Set test stack config type for api_recorder test isolation
    stack_config = session.config.getoption("--stack-config", default=None)
    if stack_config and (
        stack_config.startswith("server:") or stack_config.startswith("docker:") or stack_config.startswith("http")
    ):
        os.environ["LLAMA_STACK_TEST_STACK_CONFIG_TYPE"] = "server"
        logger.info(f"Test stack config type: server (stack_config={stack_config})")
    else:
        os.environ["LLAMA_STACK_TEST_STACK_CONFIG_TYPE"] = "library_client"
        logger.info(f"Test stack config type: library_client (stack_config={stack_config})")

    patch_httpx_for_test_id()


@pytest.fixture(autouse=True)
def suppress_httpx_logs(caplog):
    """Suppress httpx INFO logs for all integration tests"""
    caplog.set_level(logging.WARNING, logger="httpx")


@pytest.fixture(autouse=True)
def _track_test_context(request):
    """Automatically track current test context for isolated recordings.

    This fixture runs for every test and stores the test's nodeid in a contextvar
    that the recording system can access to determine which subdirectory to use.
    """
    from llama_stack.core.testing_context import reset_test_context, set_test_context

    token = set_test_context(request.node.nodeid)

    yield

    reset_test_context(token)


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

    inference_mode = config.getoption("--inference-mode")
    os.environ["LLAMA_STACK_TEST_INFERENCE_MODE"] = inference_mode

    suite = config.getoption("--suite")
    if suite:
        if suite not in SUITE_DEFINITIONS:
            raise pytest.UsageError(f"Unknown suite: {suite}. Available: {', '.join(sorted(SUITE_DEFINITIONS.keys()))}")

    # Apply setups (global parameterizations): env + defaults
    setup = config.getoption("--setup")
    if suite and not setup:
        setup = SUITE_DEFINITIONS[suite].default_setup

    if setup:
        if setup not in SETUP_DEFINITIONS:
            raise pytest.UsageError(
                f"Unknown setup '{setup}'. Available: {', '.join(sorted(SETUP_DEFINITIONS.keys()))}"
            )

        setup_obj = SETUP_DEFINITIONS[setup]
        logger.info(f"Applying setup '{setup}'{' for suite ' + suite if suite else ''}")
        # Apply env first
        for k, v in setup_obj.env.items():
            if k not in os.environ:
                os.environ[k] = str(v)
        # Apply defaults if not provided explicitly
        for dest, value in setup_obj.defaults.items():
            current = getattr(config.option, dest, None)
            if current is None:
                setattr(config.option, dest, value)

    # Apply global fallback for embedding_dimension if still not set
    if getattr(config.option, "embedding_dimension", None) is None:
        config.option.embedding_dimension = 384

    # Apply global fallback for embedding_model when using stack configs with embedding models
    if getattr(config.option, "embedding_model", None) is None:
        stack_config = config.getoption("--stack-config", default=None)
        if stack_config and "inference=inline::sentence-transformers" in stack_config:
            # Use the full qualified model ID that matches what's actually registered
            config.option.embedding_model = "inline::sentence-transformers/nomic-ai/nomic-embed-text-v1.5"


def pytest_addoption(parser):
    parser.addoption(
        "--stack-config",
        help=textwrap.dedent(
            """
            a 'pointer' to the stack. this can be either be:
            (a) a template name like `starter`, or
            (b) a path to a config.yaml file, or
            (c) an adhoc config spec, e.g. `inference=fireworks,safety=llama-guard,agents=meta-reference`, or
            (d) a server config like `server:ci-tests`, or
            (e) a docker config like `docker:ci-tests` (builds and runs container)
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
        "--rerank-model",
        help="comma-separated list of rerank models. Fixture name: rerank_model_id",
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
        default=768,
        help="Output dimensionality of the embedding model to use for testing. Default: 768",
    )

    parser.addoption(
        "--inference-mode",
        help="Inference mode: { record, replay, live, record-if-missing } (default: replay)",
        choices=["record", "replay", "live", "record-if-missing"],
        default="replay",
    )
    parser.addoption(
        "--report",
        help="Path where the test report should be written, e.g. --report=/path/to/report.md",
    )

    available_suites = ", ".join(sorted(SUITE_DEFINITIONS.keys()))
    suite_help = (
        f"Single test suite to run (narrows collection). Available: {available_suites}. Example: --suite=responses"
    )
    parser.addoption("--suite", help=suite_help)

    # Global setups for any suite
    available_setups = ", ".join(sorted(SETUP_DEFINITIONS.keys()))
    setup_help = (
        f"Global test setup configuration. Available: {available_setups}. "
        "Can be used with any suite. Example: --setup=ollama"
    )
    parser.addoption("--setup", help=setup_help)


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
    "nomic-ai/nomic-embed-text-v1.5": "Nomic-v1.5",
}


def get_short_id(value):
    return MODEL_SHORT_IDS.get(value, value)


def parse_vector_io_providers_from_config(config):
    """Parse stack config to extract vector_io provider from command line."""
    config_str = config.getoption("--stack-config", default=None) or os.environ.get("LLAMA_STACK_CONFIG")

    if not config_str:
        return None

    try:
        # Handle stack-config format: "files=inline::localfs,inference=inline::sentence-transformers,vector_io=inline::milvus"
        for part in config_str.replace(";", ",").split(","):
            part = part.strip()
            if part.startswith("vector_io="):
                provider_spec = part.split("=", 1)[1].strip()
                # Return the full provider specification (e.g. "inline::milvus")
                # The runtime system expects full provider IDs
                return [provider_spec]
    except Exception as e:
        logger.debug(f"Failed to parse vector_io provider from config: {e}")

    return None


def pytest_generate_tests(metafunc):
    """
    This is the main function which processes CLI arguments and generates various combinations of parameters.
    It is also responsible for generating test IDs which are succinct enough.

    Each option can be comma separated list of values which results in multiple parameter combinations.
    """
    # Handle vector_io_provider_id dynamically
    if "vector_io_provider_id" in metafunc.fixturenames:
        providers = parse_vector_io_providers_from_config(metafunc.config)
        if providers:
            # Use the configured provider instead of letting decorator handle it
            # Use short names in test IDs for readability
            test_ids = [f"vector_io={p.split('::')[-1] if '::' in p else p}" for p in providers]
            metafunc.parametrize("vector_io_provider_id", providers, ids=test_ids)
        else:
            # No stack config found, apply fallback parametrization here
            inference_mode = os.environ.get("LLAMA_STACK_TEST_INFERENCE_MODE")
            if inference_mode == "live":
                all_providers = ["faiss", "sqlite-vec", "milvus", "chromadb", "pgvector", "weaviate", "qdrant"]
            else:
                all_providers = ["faiss", "sqlite-vec"]

            test_ids = [f"vector_io={p.split('::')[-1] if '::' in p else p}" for p in all_providers]
            metafunc.parametrize("vector_io_provider_id", all_providers, ids=test_ids)

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
        "rerank_model_id": ("--rerank-model", "rerank"),
    }

    # Collect all parameters and their values
    for fixture_name, (option, id_prefix) in fixture_configs.items():
        if fixture_name not in metafunc.fixturenames:
            continue

        params.append(fixture_name)
        # Use getattr on config.option to see values set by pytest_configure fallbacks
        dest = option.lstrip("-").replace("-", "_")
        val = getattr(metafunc.config.option, dest, None)

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


def pytest_ignore_collect(path: str, config: pytest.Config) -> bool:
    """Skip collecting paths outside the selected suite roots for speed."""
    suite = config.getoption("--suite")
    if not suite:
        return False

    sobj = SUITE_DEFINITIONS.get(suite)
    roots: list[str] = sobj.get("roots", []) if isinstance(sobj, dict) else getattr(sobj, "roots", [])
    if not roots:
        return False

    p = Path(str(path)).resolve()

    # Only constrain within tests/integration to avoid ignoring unrelated tests
    integration_root = (Path(str(config.rootpath)) / "tests" / "integration").resolve()
    if not p.is_relative_to(integration_root):
        return False

    for r in roots:
        rp = (Path(str(config.rootpath)) / r).resolve()
        if rp.is_file():
            # Allow the exact file and any ancestor directories so pytest can walk into it.
            if p == rp:
                return False
            if p.is_dir() and rp.is_relative_to(p):
                return False
        else:
            # Allow anything inside an allowed directory
            if p.is_relative_to(rp):
                return False
    return True


def get_vector_io_provider_ids(client):
    """Get all available vector_io provider IDs."""
    providers = [p for p in client.providers.list() if p.api == "vector_io"]
    return [p.provider_id for p in providers]


def vector_provider_wrapper(func):
    """Decorator with runtime validation and fallback parametrization."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import inspect

        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        vector_io_provider_id = bound_args.arguments.get("vector_io_provider_id")
        if not vector_io_provider_id:
            pytest.skip("No vector_io_provider_id provided")

        # Get client_with_models to check available providers
        client_with_models = bound_args.arguments.get("client_with_models")
        if client_with_models:
            available_providers = get_vector_io_provider_ids(client_with_models)
            if vector_io_provider_id not in available_providers:
                pytest.skip(f"Provider '{vector_io_provider_id}' not available. Available: {available_providers}")

        return func(*args, **kwargs)

    # Always return just the wrapper - pytest_generate_tests handles parametrization
    # If pytest_generate_tests doesn't parametrize, that means there was no
    # vector_io_provider_id in fixturenames, so no parametrization is needed
    return wrapper


@pytest.fixture
def vector_io_provider_id(request, client_with_models):
    """Fixture that provides a specific vector_io provider ID, skipping if not available."""
    if hasattr(request, "param"):
        requested_provider = request.param
        available_providers = get_vector_io_provider_ids(client_with_models)

        if requested_provider not in available_providers:
            pytest.skip(f"Provider '{requested_provider}' not available. Available: {available_providers}")

        return requested_provider
    else:
        provider_ids = get_vector_io_provider_ids(client_with_models)
        if not provider_ids:
            pytest.skip("No vector_io providers available")
        return provider_ids[0]
