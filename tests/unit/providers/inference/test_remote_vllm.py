# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChoice,
    ToolChoice,
)
from llama_stack.apis.models import Model
from llama_stack.providers.datatypes import HealthStatus
from llama_stack.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter

# These are unit test for the remote vllm provider
# implementation. This should only contain tests which are specific to
# the implementation details of those classes. More general
# (API-level) tests should be placed in tests/integration/inference/
#
# How to run this test:
#
# pytest tests/unit/providers/inference/test_remote_vllm.py \
# -v -s --tb=short --disable-warnings


@pytest.fixture(scope="function")
async def vllm_inference_adapter():
    config = VLLMInferenceAdapterConfig(url="http://mocked.localhost:12345")
    inference_adapter = VLLMInferenceAdapter(config=config)
    inference_adapter.model_store = AsyncMock()
    await inference_adapter.initialize()
    return inference_adapter


async def test_old_vllm_tool_choice(vllm_inference_adapter):
    """
    Test that we set tool_choice to none when no tools are in use
    to support older versions of vLLM
    """
    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm-inference")
    vllm_inference_adapter.model_store.get_model.return_value = mock_model

    # Patch the client property to avoid instantiating a real AsyncOpenAI client
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_property:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_client_property.return_value = mock_client

        # No tools but auto tool choice
        await vllm_inference_adapter.openai_chat_completion(
            "mock-model",
            [],
            stream=False,
            tools=None,
            tool_choice=ToolChoice.auto.value,
        )
        mock_client.chat.completions.create.assert_called()
        call_args = mock_client.chat.completions.create.call_args
        # Ensure tool_choice gets converted to none for older vLLM versions
        assert call_args.kwargs["tool_choice"] == ToolChoice.none.value


async def test_health_status_success(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when the connection is successful.

    This test verifies that the health method returns a HealthResponse with status OK
    when the /health endpoint responds successfully.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Call the health method
        health_response = await vllm_inference_adapter.health()

        # Verify the response
        assert health_response["status"] == HealthStatus.OK

        # Verify that the health endpoint was called
        mock_client_instance.get.assert_called_once()
        call_args = mock_client_instance.get.call_args[0]
        assert call_args[0].endswith("/health")


async def test_health_status_failure(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when the connection fails.

    This test verifies that the health method returns a HealthResponse with status ERROR
    and an appropriate error message when the connection to the vLLM server fails.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create mock client instance that raises an exception
        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = Exception("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Call the health method
        health_response = await vllm_inference_adapter.health()

        # Verify the response
        assert health_response["status"] == HealthStatus.ERROR
        assert "Health check failed: Connection failed" in health_response["message"]


async def test_health_status_no_static_api_key(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when no static API key is provided.

    This test verifies that the health method returns a HealthResponse with status OK
    when the /health endpoint responds successfully, regardless of API token configuration.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Call the health method
        health_response = await vllm_inference_adapter.health()

        # Verify the response
        assert health_response["status"] == HealthStatus.OK


async def test_openai_chat_completion_is_async(vllm_inference_adapter):
    """
    Verify that openai_chat_completion is async and doesn't block the event loop.

    To do this we mock the underlying inference with a sleep, start multiple
    inference calls in parallel, and ensure the total time taken is less
    than the sum of the individual sleep times.
    """
    sleep_time = 0.5

    async def mock_create(*args, **kwargs):
        await asyncio.sleep(sleep_time)
        return OpenAIChatCompletion(
            id="chatcmpl-abc123",
            created=1,
            model="mock-model",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        content="nothing interesting",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
        )

    async def do_inference():
        await vllm_inference_adapter.openai_chat_completion(
            "mock-model", messages=["one fish", "two fish"], stream=False
        )

    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_create_client.return_value = mock_client

        start_time = time.time()
        await asyncio.gather(do_inference(), do_inference(), do_inference(), do_inference())
        total_time = time.time() - start_time

        assert mock_create_client.call_count == 4  # no cheating
        assert total_time < (sleep_time * 2), f"Total time taken: {total_time}s exceeded expected max"


async def test_should_refresh_models():
    """
    Test the should_refresh_models method with different refresh_models configurations.

    This test verifies that:
    1. When refresh_models is True, should_refresh_models returns True regardless of api_token
    2. When refresh_models is False, should_refresh_models returns False regardless of api_token
    """

    # Test case 1: refresh_models is True, api_token is None
    config1 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token=None, refresh_models=True)
    adapter1 = VLLMInferenceAdapter(config=config1)
    result1 = await adapter1.should_refresh_models()
    assert result1 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 2: refresh_models is True, api_token is empty string
    config2 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="", refresh_models=True)
    adapter2 = VLLMInferenceAdapter(config=config2)
    result2 = await adapter2.should_refresh_models()
    assert result2 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 3: refresh_models is True, api_token is "fake" (default)
    config3 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="fake", refresh_models=True)
    adapter3 = VLLMInferenceAdapter(config=config3)
    result3 = await adapter3.should_refresh_models()
    assert result3 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 4: refresh_models is True, api_token is real token
    config4 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="real-token-123", refresh_models=True)
    adapter4 = VLLMInferenceAdapter(config=config4)
    result4 = await adapter4.should_refresh_models()
    assert result4 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 5: refresh_models is False, api_token is real token
    config5 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="real-token-456", refresh_models=False)
    adapter5 = VLLMInferenceAdapter(config=config5)
    result5 = await adapter5.should_refresh_models()
    assert result5 is False, "should_refresh_models should return False when refresh_models is False"
