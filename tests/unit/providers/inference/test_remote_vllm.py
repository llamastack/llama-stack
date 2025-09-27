# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as OpenAIChoiceChunk,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta as OpenAIChoiceDelta,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.model import Model as OpenAIModel

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponseEventType,
    CompletionMessage,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChoice,
    SystemMessage,
    ToolChoice,
    ToolConfig,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.models import Model
from llama_stack.models.llama.datatypes import StopReason, ToolCall
from llama_stack.providers.datatypes import HealthStatus
from llama_stack.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.inference.vllm.vllm import (
    VLLMInferenceAdapter,
    _process_vllm_chat_completion_stream_response,
)

# These are unit test for the remote vllm provider
# implementation. This should only contain tests which are specific to
# the implementation details of those classes. More general
# (API-level) tests should be placed in tests/integration/inference/
#
# How to run this test:
#
# pytest tests/unit/providers/inference/test_remote_vllm.py \
# -v -s --tb=short --disable-warnings


@pytest.fixture(scope="module")
def mock_openai_models_list():
    with patch("openai.resources.models.AsyncModels.list") as mock_list:
        yield mock_list


@pytest.fixture(scope="function")
async def vllm_inference_adapter():
    config = VLLMInferenceAdapterConfig(url="http://mocked.localhost:12345")
    inference_adapter = VLLMInferenceAdapter(config)
    inference_adapter.model_store = AsyncMock()
    # Mock the __provider_spec__ attribute that would normally be set by the resolver
    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_type = "vllm-inference"
    inference_adapter.__provider_spec__.provider_data_validator = MagicMock()
    await inference_adapter.initialize()
    return inference_adapter


async def test_register_model_checks_vllm(mock_openai_models_list, vllm_inference_adapter):
    async def mock_openai_models():
        yield OpenAIModel(id="foo", created=1, object="model", owned_by="test")

    mock_openai_models_list.return_value = mock_openai_models()

    foo_model = Model(identifier="foo", provider_resource_id="foo", provider_id="vllm-inference")

    await vllm_inference_adapter.register_model(foo_model)
    mock_openai_models_list.assert_called()


async def test_old_vllm_tool_choice(vllm_inference_adapter):
    """
    Test that we set tool_choice to none when no tools are in use
    to support older versions of vLLM
    """
    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm-inference")
    vllm_inference_adapter.model_store.get_model.return_value = mock_model

    with patch.object(vllm_inference_adapter, "_nonstream_chat_completion") as mock_nonstream_completion:
        # No tools but auto tool choice
        await vllm_inference_adapter.chat_completion(
            "mock-model",
            [],
            stream=False,
            tools=None,
            tool_config=ToolConfig(tool_choice=ToolChoice.auto),
        )
        mock_nonstream_completion.assert_called()
        request = mock_nonstream_completion.call_args.args[0]
        # Ensure tool_choice gets converted to none for older vLLM versions
        assert request.tool_config.tool_choice == ToolChoice.none


async def test_tool_call_response(vllm_inference_adapter):
    """Verify that tool call arguments from a CompletionMessage are correctly converted
    into the expected JSON format."""

    # Patch the client property to avoid instantiating a real AsyncOpenAI client
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_create_client.return_value = mock_client

        # Mock the model to return a proper provider_resource_id
        mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm-inference")
        vllm_inference_adapter.model_store.get_model.return_value = mock_model

        messages = [
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="How many?"),
            CompletionMessage(
                content="",
                stop_reason=StopReason.end_of_turn,
                tool_calls=[
                    ToolCall(
                        call_id="foo",
                        tool_name="knowledge_search",
                        arguments={"query": "How many?"},
                        arguments_json='{"query": "How many?"}',
                    )
                ],
            ),
            ToolResponseMessage(call_id="foo", content="knowledge_search found 5...."),
        ]
        await vllm_inference_adapter.chat_completion(
            "mock-model",
            messages,
            stream=False,
            tools=[],
            tool_config=ToolConfig(tool_choice=ToolChoice.auto),
        )

        assert mock_client.chat.completions.create.call_args.kwargs["messages"][2]["tool_calls"] == [
            {
                "id": "foo",
                "type": "function",
                "function": {"name": "knowledge_search", "arguments": '{"query": "How many?"}'},
            }
        ]


async def test_tool_call_delta_empty_tool_call_buf():
    """
    Test that we don't generate extra chunks when processing a
    tool call response that didn't call any tools. Previously we would
    emit chunks with spurious ToolCallParseStatus.succeeded or
    ToolCallParseStatus.failed when processing chunks that didn't
    actually make any tool calls.
    """

    async def mock_stream():
        delta = OpenAIChoiceDelta(content="", tool_calls=None)
        choices = [OpenAIChoiceChunk(delta=delta, finish_reason="stop", index=0)]
        mock_chunk = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=choices,
        )
        for chunk in [mock_chunk]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 2
    assert chunks[0].event.event_type.value == "start"
    assert chunks[1].event.event_type.value == "complete"
    assert chunks[1].event.stop_reason == StopReason.end_of_turn


async def test_tool_call_delta_streaming_arguments_dict():
    async def mock_stream():
        mock_chunk_1 = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="tc_1",
                                index=1,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="power",
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_2 = OpenAIChatCompletionChunk(
            id="chunk-2",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="tc_1",
                                index=1,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="power",
                                    arguments='{"number": 28, "power": 3}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_3 = OpenAIChatCompletionChunk(
            id="chunk-3",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(content="", tool_calls=None), finish_reason="tool_calls", index=0
                )
            ],
        )
        for chunk in [mock_chunk_1, mock_chunk_2, mock_chunk_3]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[0].event.event_type.value == "start"
    assert chunks[1].event.event_type.value == "progress"
    assert chunks[1].event.delta.type == "tool_call"
    assert chunks[1].event.delta.parse_status.value == "succeeded"
    assert chunks[1].event.delta.tool_call.arguments_json == '{"number": 28, "power": 3}'
    assert chunks[2].event.event_type.value == "complete"


async def test_multiple_tool_calls():
    async def mock_stream():
        mock_chunk_1 = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="",
                                index=1,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="power",
                                    arguments='{"number": 28, "power": 3}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_2 = OpenAIChatCompletionChunk(
            id="chunk-2",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="",
                                index=2,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="multiple",
                                    arguments='{"first_number": 4, "second_number": 7}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_3 = OpenAIChatCompletionChunk(
            id="chunk-3",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(content="", tool_calls=None), finish_reason="tool_calls", index=0
                )
            ],
        )
        for chunk in [mock_chunk_1, mock_chunk_2, mock_chunk_3]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 4
    assert chunks[0].event.event_type.value == "start"
    assert chunks[1].event.event_type.value == "progress"
    assert chunks[1].event.delta.type == "tool_call"
    assert chunks[1].event.delta.parse_status.value == "succeeded"
    assert chunks[1].event.delta.tool_call.arguments_json == '{"number": 28, "power": 3}'
    assert chunks[2].event.event_type.value == "progress"
    assert chunks[2].event.delta.type == "tool_call"
    assert chunks[2].event.delta.parse_status.value == "succeeded"
    assert chunks[2].event.delta.tool_call.arguments_json == '{"first_number": 4, "second_number": 7}'
    assert chunks[3].event.event_type.value == "complete"


async def test_process_vllm_chat_completion_stream_response_no_choices():
    """
    Test that we don't error out when vLLM returns no choices for a
    completion request. This can happen when there's an error thrown
    in vLLM for example.
    """

    async def mock_stream():
        choices = []
        mock_chunk = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=choices,
        )
        for chunk in [mock_chunk]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 1
    assert chunks[0].event.event_type.value == "start"


async def test_get_params_empty_tools(vllm_inference_adapter):
    request = ChatCompletionRequest(
        tools=[],
        model="test_model",
        messages=[UserMessage(content="test")],
    )
    params = await vllm_inference_adapter._get_params(request)
    assert "tools" not in params


async def test_process_vllm_chat_completion_stream_response_tool_call_args_last_chunk():
    """
    Tests the edge case where the model returns the arguments for the tool call in the same chunk that
    contains the finish reason (i.e., the last one).
    We want to make sure the tool call is executed in this case, and the parameters are passed correctly.
    """

    mock_tool_name = "mock_tool"
    mock_tool_arguments = {"arg1": 0, "arg2": 100}
    mock_tool_arguments_str = json.dumps(mock_tool_arguments)

    async def mock_stream():
        mock_chunks = [
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "mock_id",
                                    "type": "function",
                                    "function": {
                                        "name": mock_tool_name,
                                        "arguments": None,
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": None,
                                    "function": {
                                        "name": None,
                                        "arguments": mock_tool_arguments_str,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        for chunk in mock_chunks:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[-1].event.event_type == ChatCompletionResponseEventType.complete
    assert chunks[-2].event.delta.type == "tool_call"
    assert chunks[-2].event.delta.tool_call.tool_name == mock_tool_name
    assert chunks[-2].event.delta.tool_call.arguments == mock_tool_arguments


async def test_process_vllm_chat_completion_stream_response_no_finish_reason():
    """
    Tests the edge case where the model requests a tool call and stays idle without explicitly providing the
    finish reason.
    We want to make sure that this case is recognized and handled correctly, i.e., as a valid end of message.
    """

    mock_tool_name = "mock_tool"
    mock_tool_arguments = {"arg1": 0, "arg2": 100}
    mock_tool_arguments_str = '"{\\"arg1\\": 0, \\"arg2\\": 100}"'

    async def mock_stream():
        mock_chunks = [
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "mock_id",
                                    "type": "function",
                                    "function": {
                                        "name": mock_tool_name,
                                        "arguments": mock_tool_arguments_str,
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        for chunk in mock_chunks:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[-1].event.event_type == ChatCompletionResponseEventType.complete
    assert chunks[-2].event.delta.type == "tool_call"
    assert chunks[-2].event.delta.tool_call.tool_name == mock_tool_name
    assert chunks[-2].event.delta.tool_call.arguments == mock_tool_arguments


async def test_process_vllm_chat_completion_stream_response_tool_without_args():
    """
    Tests the edge case where no arguments are provided for the tool call.
    Tool calls with no arguments should be treated as regular tool calls, which was not the case until now.
    """
    mock_tool_name = "mock_tool"

    async def mock_stream():
        mock_chunks = [
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "mock_id",
                                    "type": "function",
                                    "function": {
                                        "name": mock_tool_name,
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        for chunk in mock_chunks:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[-1].event.event_type == ChatCompletionResponseEventType.complete
    assert chunks[-2].event.delta.type == "tool_call"
    assert chunks[-2].event.delta.tool_call.tool_name == mock_tool_name
    assert chunks[-2].event.delta.tool_call.arguments == {}


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
    adapter1 = VLLMInferenceAdapter(config1)
    result1 = await adapter1.should_refresh_models()
    assert result1 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 2: refresh_models is True, api_token is empty string
    config2 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="", refresh_models=True)
    adapter2 = VLLMInferenceAdapter(config2)
    result2 = await adapter2.should_refresh_models()
    assert result2 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 3: refresh_models is True, api_token is "fake" (default)
    config3 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="fake", refresh_models=True)
    adapter3 = VLLMInferenceAdapter(config3)
    result3 = await adapter3.should_refresh_models()
    assert result3 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 4: refresh_models is True, api_token is real token
    config4 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="real-token-123", refresh_models=True)
    adapter4 = VLLMInferenceAdapter(config4)
    result4 = await adapter4.should_refresh_models()
    assert result4 is True, "should_refresh_models should return True when refresh_models is True"

    # Test case 5: refresh_models is False, api_token is real token
    config5 = VLLMInferenceAdapterConfig(url="http://test.localhost", api_token="real-token-456", refresh_models=False)
    adapter5 = VLLMInferenceAdapter(config5)
    result5 = await adapter5.should_refresh_models()
    assert result5 is False, "should_refresh_models should return False when refresh_models is False"


async def test_provider_data_var_context_propagation(vllm_inference_adapter):
    """
    Test that PROVIDER_DATA_VAR context is properly propagated through the vLLM inference adapter.
    This ensures that dynamic provider data (like API tokens) can be passed through context.
    Note: The base URL is always taken from config.url, not from provider data.
    """
    # Mock the AsyncOpenAI class to capture provider data
    with (
        patch("llama_stack.providers.utils.inference.openai_mixin.AsyncOpenAI") as mock_openai_class,
        patch.object(vllm_inference_adapter, "get_request_provider_data") as mock_get_provider_data,
    ):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock provider data to return test data
        mock_provider_data = MagicMock()
        mock_provider_data.vllm_api_token = "test-token-123"
        mock_provider_data.vllm_url = "http://test-server:8000/v1"
        mock_get_provider_data.return_value = mock_provider_data

        # Mock the model
        mock_model = Model(identifier="test-model", provider_resource_id="test-model", provider_id="vllm-inference")
        vllm_inference_adapter.model_store.get_model.return_value = mock_model

        try:
            # Execute chat completion
            await vllm_inference_adapter.chat_completion(
                "test-model",
                [UserMessage(content="Hello")],
                stream=False,
                tools=None,
                tool_config=ToolConfig(tool_choice=ToolChoice.auto),
            )

            # Verify that ALL client calls were made with the correct parameters
            calls = mock_openai_class.call_args_list
            incorrect_calls = []

            for i, call in enumerate(calls):
                api_key = call[1]["api_key"]
                base_url = call[1]["base_url"]

                if api_key != "test-token-123" or base_url != "http://mocked.localhost:12345":
                    incorrect_calls.append({"call_index": i, "api_key": api_key, "base_url": base_url})

            if incorrect_calls:
                error_msg = (
                    f"Found {len(incorrect_calls)} calls with incorrect parameters out of {len(calls)} total calls:\n"
                )
                for incorrect_call in incorrect_calls:
                    error_msg += f"  Call {incorrect_call['call_index']}: api_key='{incorrect_call['api_key']}', base_url='{incorrect_call['base_url']}'\n"
                error_msg += "Expected: api_key='test-token-123', base_url='http://mocked.localhost:12345'"
                raise AssertionError(error_msg)

            # Ensure at least one call was made
            assert len(calls) >= 1, "No AsyncOpenAI client calls were made"

            # Verify that chat completion was called
            mock_client.chat.completions.create.assert_called_once()

        finally:
            # Clean up context
            pass
