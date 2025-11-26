# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Fixtures for MLflow integration tests.

These tests require a running MLflow server. Set the MLFLOW_TRACKING_URI
environment variable to point to your MLflow server, or the tests will
attempt to use http://localhost:5555.

To run tests:
    # Start MLflow server (in separate terminal)
    mlflow server --host 127.0.0.1 --port 5555

    # Run integration tests
    MLFLOW_TRACKING_URI=http://localhost:5555 \
        uv run --group test pytest -sv tests/integration/providers/remote/prompts/mlflow/
"""

import os

import pytest

from llama_stack.providers.remote.prompts.mlflow import MLflowPromptsAdapter
from llama_stack.providers.remote.prompts.mlflow.config import MLflowPromptsConfig


@pytest.fixture(scope="session")
def mlflow_tracking_uri():
    """Get MLflow tracking URI from environment or use default."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")


@pytest.fixture(scope="session")
def mlflow_server_available(mlflow_tracking_uri):
    """Verify MLflow server is running and accessible.

    Skips all tests if server is not available.
    """
    try:
        import requests

        response = requests.get(f"{mlflow_tracking_uri}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"MLflow server at {mlflow_tracking_uri} returned status {response.status_code}")
    except ImportError:
        pytest.skip("requests package not installed - install with: pip install requests")
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"MLflow server not available at {mlflow_tracking_uri}. "
            "Start with: mlflow server --host 127.0.0.1 --port 5555"
        )
    except requests.exceptions.Timeout:
        pytest.skip(f"MLflow server at {mlflow_tracking_uri} timed out")
    except Exception as e:
        pytest.skip(f"Failed to check MLflow server availability: {e}")

    return True


@pytest.fixture
async def mlflow_config(mlflow_tracking_uri, mlflow_server_available):
    """Create MLflow configuration for testing."""
    return MLflowPromptsConfig(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name="test-llama-stack-prompts",
        timeout_seconds=30,
    )


@pytest.fixture
async def mlflow_adapter(mlflow_config):
    """Create and initialize MLflow adapter for testing.

    This fixture creates a new adapter instance for each test.
    The adapter connects to the MLflow server specified in the config.
    """
    adapter = MLflowPromptsAdapter(config=mlflow_config)
    await adapter.initialize()

    yield adapter

    # Cleanup: shutdown adapter
    await adapter.shutdown()


@pytest.fixture
async def mlflow_adapter_with_cleanup(mlflow_config):
    """Create MLflow adapter with automatic cleanup after test.

    This fixture is useful for tests that create prompts and want them
    automatically cleaned up (though MLflow doesn't support deletion,
    so cleanup is best-effort).
    """
    adapter = MLflowPromptsAdapter(config=mlflow_config)
    await adapter.initialize()

    created_prompt_ids = []

    # Provide adapter and tracking list
    class AdapterWithTracking:
        def __init__(self, adapter_instance):
            self.adapter = adapter_instance
            self.created_ids = created_prompt_ids

        async def create_prompt(self, *args, **kwargs):
            prompt = await self.adapter.create_prompt(*args, **kwargs)
            self.created_ids.append(prompt.prompt_id)
            return prompt

        def __getattr__(self, name):
            return getattr(self.adapter, name)

    tracked_adapter = AdapterWithTracking(adapter)

    yield tracked_adapter

    # Cleanup: attempt to delete created prompts
    # Note: MLflow doesn't support deletion, so this is a no-op
    # but we keep it for future compatibility
    for prompt_id in created_prompt_ids:
        try:
            await adapter.delete_prompt(prompt_id)
        except NotImplementedError:
            # Expected - MLflow doesn't support deletion
            pass
        except Exception:
            # Ignore cleanup errors
            pass

    await adapter.shutdown()
