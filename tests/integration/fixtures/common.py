# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import pytest
import requests
import yaml
from llama_stack_client import LlamaStackClient
from openai import OpenAI

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.core.library_client import LlamaStackAsLibraryClient
from llama_stack.core.stack import run_config_from_adhoc_config_spec
from llama_stack.env import get_env_or_fail

DEFAULT_PORT = 8321


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False


def start_llama_stack_server(config_name: str) -> subprocess.Popen:
    """Start a llama stack server with the given config."""

    # remove server.log if it exists
    if os.path.exists("server.log"):
        os.remove("server.log")

    cmd = f"llama stack run {config_name}"
    devnull = open(os.devnull, "w")
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=devnull,  # redirect stdout to devnull to prevent deadlock
        stderr=subprocess.PIPE,  # keep stderr to see errors
        text=True,
        env={**os.environ, "LLAMA_STACK_LOG_FILE": "server.log"},
        # Create new process group so we can kill all child processes
        preexec_fn=os.setsid,
    )
    return process


def wait_for_server_ready(base_url: str, timeout: int = 30, process: subprocess.Popen | None = None) -> bool:
    """Wait for the server to be ready by polling the health endpoint."""
    health_url = f"{base_url}/v1/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        if process and process.poll() is not None:
            print(f"Server process terminated with return code: {process.returncode}")
            print(f"Server stderr: {process.stderr.read()}")
            return False

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass

        # Print progress every 5 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 5 == 0 and elapsed > 0:
            print(f"Waiting for server at {base_url}... ({elapsed:.1f}s elapsed)")

        time.sleep(0.5)

    print(f"Server failed to respond within {timeout} seconds")
    return False


def stop_server_on_port(port: int, timeout: float = 10.0) -> None:
    """Terminate any server processes bound to the given port."""

    try:
        output = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    pids = {int(line) for line in output.splitlines() if line.strip()}
    if not pids:
        return

    deadline = time.time() + timeout
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in list(pids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pids.discard(pid)

        while not is_port_available(port) and time.time() < deadline:
            time.sleep(0.1)

        if is_port_available(port):
            return

    raise RuntimeError(f"Unable to free port {port} for test server restart")


def get_provider_data():
    # TODO: this needs to be generalized so each provider can have a sample provider data just
    # like sample run config on which we can do replace_env_vars()
    keymap = {
        "TAVILY_SEARCH_API_KEY": "tavily_search_api_key",
        "BRAVE_SEARCH_API_KEY": "brave_search_api_key",
        "FIREWORKS_API_KEY": "fireworks_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "GROQ_API_KEY": "groq_api_key",
        "WOLFRAM_ALPHA_API_KEY": "wolfram_alpha_api_key",
    }
    provider_data = {}
    for key, value in keymap.items():
        if os.environ.get(key):
            provider_data[value] = os.environ[key]
    return provider_data


@pytest.fixture(scope="session")
def inference_provider_type(llama_stack_client):
    providers = llama_stack_client.providers.list()
    inference_providers = [p for p in providers if p.api == "inference"]
    assert len(inference_providers) > 0, "No inference providers found"
    return inference_providers[0].provider_type


@pytest.fixture(scope="session")
def client_with_models(
    llama_stack_client,
    text_model_id,
    vision_model_id,
    embedding_model_id,
    judge_model_id,
    rerank_model_id,
):
    client = llama_stack_client

    providers = [p for p in client.providers.list() if p.api == "inference"]
    assert len(providers) > 0, "No inference providers found"

    model_ids = {m.id for m in client.models.list()}

    if text_model_id and text_model_id not in model_ids:
        raise ValueError(f"text_model_id {text_model_id} not found")
    if vision_model_id and vision_model_id not in model_ids:
        raise ValueError(f"vision_model_id {vision_model_id} not found")
    if judge_model_id and judge_model_id not in model_ids:
        raise ValueError(f"judge_model_id {judge_model_id} not found")

    if embedding_model_id and embedding_model_id not in model_ids:
        raise ValueError(f"embedding_model_id {embedding_model_id} not found")

    if rerank_model_id and rerank_model_id not in model_ids:
        raise ValueError(f"rerank_model_id {rerank_model_id} not found")
    return client


@pytest.fixture(scope="session")
def available_shields(llama_stack_client):
    return [shield.identifier for shield in llama_stack_client.shields.list()]


@pytest.fixture(scope="session")
def model_providers(llama_stack_client):
    return {x.provider_id for x in llama_stack_client.providers.list() if x.api == "inference"}


@pytest.fixture(autouse=True)
def skip_if_no_model(request):
    model_fixtures = [
        "text_model_id",
        "vision_model_id",
        "embedding_model_id",
        "judge_model_id",
        "shield_id",
        "rerank_model_id",
    ]
    test_func = request.node.function

    actual_params = inspect.signature(test_func).parameters.keys()
    for fixture in model_fixtures:
        # Only check fixtures that are actually in the test function's signature
        if fixture in actual_params and fixture in request.fixturenames and not request.getfixturevalue(fixture):
            pytest.skip(f"{fixture} empty - skipping test")


@pytest.fixture(scope="session")
def llama_stack_client(request):
    # ideally, we could do this in session start given all the complex logs during initialization
    # don't clobber the test one-liner outputs. however, this also means all tests in a sub-directory
    # would be forced to use llama_stack_client, which is not what we want.
    print("\ninstantiating llama_stack_client")
    start_time = time.time()

    # Patch httpx to inject test ID for server-mode test isolation
    from llama_stack.testing.api_recorder import patch_httpx_for_test_id

    patch_httpx_for_test_id()

    client = instantiate_llama_stack_client(request.session)
    print(f"llama_stack_client instantiated in {time.time() - start_time:.3f}s")
    return client


def instantiate_llama_stack_client(session):
    config = session.config.getoption("--stack-config")
    if not config:
        config = get_env_or_fail("LLAMA_STACK_CONFIG")

    if not config:
        raise ValueError("You must specify either --stack-config or LLAMA_STACK_CONFIG")

    # Handle server:<config_name> format or server:<config_name>:<port>
    if config.startswith("server:"):
        parts = config.split(":")
        config_name = parts[1]
        port = int(parts[2]) if len(parts) > 2 else int(os.environ.get("LLAMA_STACK_PORT", DEFAULT_PORT))
        base_url = f"http://localhost:{port}"

        force_restart = os.environ.get("LLAMA_STACK_TEST_FORCE_SERVER_RESTART") == "1"
        if force_restart:
            print(f"Forcing restart of the server on port {port}")
            stop_server_on_port(port)

        # Check if port is available
        if is_port_available(port):
            print(f"Starting llama stack server with config '{config_name}' on port {port}...")

            # Start server
            server_process = start_llama_stack_server(config_name)

            # Wait for server to be ready
            if not wait_for_server_ready(base_url, timeout=120, process=server_process):
                print("Server failed to start within timeout")
                server_process.terminate()
                raise RuntimeError(
                    f"Server failed to start within timeout. Check that config '{config_name}' exists and is valid. "
                    f"See server.log for details."
                )

            print(f"Server is ready at {base_url}")

            # Store process for potential cleanup (pytest will handle termination at session end)
            session._llama_stack_server_process = server_process
        else:
            print(f"Port {port} is already in use, assuming server is already running...")

        return LlamaStackClient(
            base_url=base_url,
            provider_data=get_provider_data(),
            timeout=int(os.environ.get("LLAMA_STACK_CLIENT_TIMEOUT", "30")),
        )

    # check if this looks like a URL using proper URL parsing
    try:
        parsed_url = urlparse(config)
        if parsed_url.scheme and parsed_url.netloc:
            return LlamaStackClient(
                base_url=config,
                provider_data=get_provider_data(),
            )
    except Exception:
        # If URL parsing fails, treat as non-URL config
        pass

    if "=" in config:
        run_config = run_config_from_adhoc_config_spec(config)

        # --stack-config bypasses template so need this to set default embedding model
        if "vector_io" in config and "inference" in config:
            run_config.vector_stores = VectorStoresConfig(
                embedding_model_id="inline::sentence-transformers/nomic-ai/nomic-embed-text-v1.5"
            )

        run_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(run_config_file.name, "w") as f:
            yaml.dump(run_config.model_dump(mode="json"), f)
        config = run_config_file.name

    client = LlamaStackAsLibraryClient(
        config,
        provider_data=get_provider_data(),
        skip_logger_removal=True,
    )
    return client


@pytest.fixture(scope="session")
def require_server(llama_stack_client):
    """
    Skip test if no server is running.

    We use the llama_stack_client to tell if a server was started or not.

    We use this with openai_client because it relies on a running server.
    """
    if isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        pytest.skip("No server running")


@pytest.fixture(scope="session")
def openai_client(llama_stack_client, require_server):
    base_url = f"{llama_stack_client.base_url}/v1"
    return OpenAI(base_url=base_url, api_key="fake")


@pytest.fixture(params=["openai_client", "client_with_models"])
def compat_client(request, client_with_models):
    if request.param == "openai_client" and isinstance(client_with_models, LlamaStackAsLibraryClient):
        # OpenAI client expects a server, so unless we also rewrite OpenAI client's requests
        # to go via the Stack library client (which itself rewrites requests to be served inline),
        # we cannot do this.
        #
        # This means when we are using Stack as a library, we will test only via the Llama Stack client.
        # When we are using a server setup, we can exercise both OpenAI and Llama Stack clients.
        pytest.skip("(OpenAI) Compat client cannot be used with Stack library client")

    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session", autouse=True)
def cleanup_server_process(request):
    """Cleanup server process at the end of the test session."""
    yield  # Run tests

    if hasattr(request.session, "_llama_stack_server_process"):
        server_process = request.session._llama_stack_server_process
        if server_process:
            if server_process.poll() is None:
                print("Terminating llama stack server process...")
            else:
                print(f"Server process already terminated with return code: {server_process.returncode}")
                return
            try:
                print(f"Terminating process {server_process.pid} and its group...")
                # Kill the entire process group
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=10)
                print("Server process and children terminated gracefully")
            except subprocess.TimeoutExpired:
                print("Server process did not terminate gracefully, killing it")
                # Force kill the entire process group
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                server_process.wait()
                print("Server process and children killed")
            except Exception as e:
                print(f"Error during server cleanup: {e}")
        else:
            print("Server process not found - won't be able to cleanup")
