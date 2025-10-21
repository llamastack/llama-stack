# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from llama_stack.apis.agents import Order
from llama_stack.apis.agents.openai_responses import (
    ListOpenAIResponseInputItem,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
    WebSearchToolTypes,
)
from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIDeveloperMessageParam,
    OpenAIJSONSchema,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIUserMessageParam,
)
from llama_stack.apis.prompts import Prompt
from llama_stack.apis.tools.tools import ListToolDefsResponse, ToolDef, ToolGroups, ToolInvocationResult, ToolRuntime
from llama_stack.core.access_control.access_control import default_policy
from llama_stack.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.utils.responses.responses_store import (
    ResponsesStore,
    _OpenAIResponseObjectWithInputAndMessages,
)
from llama_stack.providers.utils.sqlstore.sqlstore import register_sqlstore_backends
from tests.unit.providers.agents.meta_reference.fixtures import load_chat_completion_fixture


@pytest.fixture
def mock_inference_api():
    inference_api = AsyncMock()
    return inference_api


@pytest.fixture
def mock_tool_groups_api():
    tool_groups_api = AsyncMock(spec=ToolGroups)
    return tool_groups_api


@pytest.fixture
def mock_tool_runtime_api():
    tool_runtime_api = AsyncMock(spec=ToolRuntime)
    return tool_runtime_api


@pytest.fixture
def mock_responses_store():
    responses_store = AsyncMock(spec=ResponsesStore)
    return responses_store


@pytest.fixture
def mock_vector_io_api():
    vector_io_api = AsyncMock()
    return vector_io_api


@pytest.fixture
def mock_conversations_api():
    """Mock conversations API for testing."""
    mock_api = AsyncMock()
    return mock_api


@pytest.fixture
def mock_safety_api():
    safety_api = AsyncMock()
    return safety_api


@pytest.fixture
def mock_prompts_api():
    prompts_api = AsyncMock()
    return prompts_api


@pytest.fixture
def mock_files_api():
    """Mock files API for testing."""
    files_api = AsyncMock()
    return files_api


@pytest.fixture
def openai_responses_impl(
    mock_inference_api,
    mock_tool_groups_api,
    mock_tool_runtime_api,
    mock_responses_store,
    mock_vector_io_api,
    mock_safety_api,
    mock_conversations_api,
    mock_prompts_api,
    mock_files_api,
):
    return OpenAIResponsesImpl(
        inference_api=mock_inference_api,
        tool_groups_api=mock_tool_groups_api,
        tool_runtime_api=mock_tool_runtime_api,
        responses_store=mock_responses_store,
        vector_io_api=mock_vector_io_api,
        safety_api=mock_safety_api,
        conversations_api=mock_conversations_api,
        prompts_api=mock_prompts_api,
        files_api=mock_files_api,
    )


async def fake_stream(fixture: str = "simple_chat_completion.yaml"):
    value = load_chat_completion_fixture(fixture)
    yield ChatCompletionChunk(
        id=value.id,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    content=c.message.content,
                    role=c.message.role,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id=t.id,
                            function=ChoiceDeltaToolCallFunction(
                                name=t.function.name,
                                arguments=t.function.arguments,
                            ),
                        )
                        for t in (c.message.tool_calls or [])
                    ],
                ),
            )
            for c in value.choices
        ],
        created=1,
        model=value.model,
        object="chat.completion.chunk",
    )


async def test_create_openai_response_with_string_input(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a simple string input."""
    # Setup
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Load the chat completion fixture
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        temperature=0.1,
        stream=True,  # Enable streaming to test content part events
    )

    # For streaming response, collect all chunks
    chunks = [chunk async for chunk in result]

    mock_inference_api.openai_chat_completion.assert_called_once_with(
        OpenAIChatCompletionRequestWithExtraBody(
            model=model,
            messages=[OpenAIUserMessageParam(role="user", content="What is the capital of Ireland?", name=None)],
            response_format=None,
            tools=None,
            stream=True,
            temperature=0.1,
            stream_options={
                "include_usage": True,
            },
        )
    )

    # Should have content part events for text streaming
    # Expected: response.created, response.in_progress, content_part.added, output_text.delta, content_part.done, response.completed
    assert len(chunks) >= 5
    assert chunks[0].type == "response.created"
    assert any(chunk.type == "response.in_progress" for chunk in chunks)

    # Check for content part events
    content_part_added_events = [c for c in chunks if c.type == "response.content_part.added"]
    content_part_done_events = [c for c in chunks if c.type == "response.content_part.done"]
    text_delta_events = [c for c in chunks if c.type == "response.output_text.delta"]

    assert len(content_part_added_events) >= 1, "Should have content_part.added event for text"
    assert len(content_part_done_events) >= 1, "Should have content_part.done event for text"
    assert len(text_delta_events) >= 1, "Should have text delta events"

    added_event = content_part_added_events[0]
    done_event = content_part_done_events[0]
    assert added_event.content_index == 0
    assert done_event.content_index == 0
    assert added_event.output_index == done_event.output_index == 0
    assert added_event.item_id == done_event.item_id
    assert added_event.response_id == done_event.response_id

    # Verify final event is completion
    assert chunks[-1].type == "response.completed"

    # When streaming, the final response is in the last chunk
    final_response = chunks[-1].response
    assert final_response.model == model
    assert len(final_response.output) == 1
    assert isinstance(final_response.output[0], OpenAIResponseMessage)
    assert final_response.output[0].id == added_event.item_id
    assert final_response.id == added_event.response_id

    openai_responses_impl.responses_store.store_response_object.assert_called_once()
    assert final_response.output[0].content[0].text == "Dublin"


async def test_create_openai_response_with_string_input_with_tools(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a simple string input and tools."""
    # Setup
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The query to search for"}},
            "required": ["query"],
        },
    )

    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Dublin",
    )

    # Execute
    for tool_name in WebSearchToolTypes:
        # Reset mock states as we loop through each tool type
        mock_inference_api.openai_chat_completion.side_effect = [
            fake_stream("tool_call_completion.yaml"),
            fake_stream(),
        ]
        openai_responses_impl.tool_groups_api.get_tool.reset_mock()
        openai_responses_impl.tool_runtime_api.invoke_tool.reset_mock()
        openai_responses_impl.responses_store.store_response_object.reset_mock()

        result = await openai_responses_impl.create_openai_response(
            input=input_text,
            model=model,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolWebSearch(
                    name=tool_name,
                )
            ],
        )

        # Verify
        first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
        first_params = first_call.args[0]
        assert first_params.messages[0].content == "What is the capital of Ireland?"
        assert first_params.tools is not None
        assert first_params.temperature == 0.1

        second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
        second_params = second_call.args[0]
        assert second_params.messages[-1].content == "Dublin"
        assert second_params.temperature == 0.1

        openai_responses_impl.tool_groups_api.get_tool.assert_called_once_with("web_search")
        openai_responses_impl.tool_runtime_api.invoke_tool.assert_called_once_with(
            tool_name="web_search",
            kwargs={"query": "What is the capital of Ireland?"},
        )

        openai_responses_impl.responses_store.store_response_object.assert_called_once()

        # Check that we got the content from our mocked tool execution result
        assert len(result.output) >= 1
        assert isinstance(result.output[1], OpenAIResponseMessage)
        assert result.output[1].content[0].text == "Dublin"
        assert result.output[1].content[0].annotations == []


async def test_create_openai_response_with_tool_call_type_none(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a tool call response that has a type of None."""
    # Setup
    input_text = "How hot it is in San Francisco today?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_toolcall():
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc_123",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments="{}"),
                                type=None,
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=True,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_weather",
                description="Get current temperature for a given location.",
                parameters={
                    "location": "string",
                },
            )
        ],
    )

    # Check that we got the content from our mocked tool execution result
    chunks = [chunk async for chunk in result]

    # Verify event types
    # Should have: response.created, response.in_progress, output_item.added,
    # function_call_arguments.delta, function_call_arguments.done, output_item.done, response.completed
    assert len(chunks) == 7

    event_types = [chunk.type for chunk in chunks]
    assert event_types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    # Verify inference API was called correctly (after iterating over result)
    first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
    first_params = first_call.args[0]
    assert first_params.messages[0].content == input_text
    assert first_params.tools is not None
    assert first_params.temperature == 0.1

    # Check response.created event (should have empty output)
    assert len(chunks[0].response.output) == 0

    # Check response.completed event (should have the tool call)
    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert len(completed_chunk.response.output) == 1
    assert completed_chunk.response.output[0].type == "function_call"
    assert completed_chunk.response.output[0].name == "get_weather"


async def test_create_openai_response_with_tool_call_function_arguments_none(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with tool calls that omit arguments."""

    input_text = "What is the time right now?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_toolcall():
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc_123",
                                function=ChoiceDeltaToolCallFunction(name="get_current_time", arguments=None),
                                type=None,
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    def assert_common_expectations(chunks) -> None:
        first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
        first_params = first_call.args[0]
        assert first_params.messages[0].content == input_text
        assert first_params.tools is not None
        assert first_params.temperature == 0.1
        assert len(chunks[0].response.output) == 0
        completed_chunk = chunks[-1]
        assert completed_chunk.type == "response.completed"
        assert len(completed_chunk.response.output) == 1
        assert completed_chunk.response.output[0].type == "function_call"
        assert completed_chunk.response.output[0].name == "get_current_time"
        assert completed_chunk.response.output[0].arguments == "{}"

    # Function does not accept arguments
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=True,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_current_time", description="Get current time for system's timezone", parameters={}
            )
        ],
    )
    chunks = [chunk async for chunk in result]
    assert [chunk.type for chunk in chunks] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert_common_expectations(chunks)

    # Function accepts optional arguments
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=True,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_current_time",
                description="Get current time for system's timezone",
                parameters={"timezone": "string"},
            )
        ],
    )
    chunks = [chunk async for chunk in result]
    assert [chunk.type for chunk in chunks] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert_common_expectations(chunks)

    # Function accepts optional arguments with additional optional fields
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=True,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_current_time",
                description="Get current time for system's timezone",
                parameters={"timezone": "string", "location": "string"},
            )
        ],
    )
    chunks = [chunk async for chunk in result]
    assert [chunk.type for chunk in chunks] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert_common_expectations(chunks)
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()


async def test_create_openai_response_with_multiple_messages(openai_responses_impl, mock_inference_api, mock_files_api):
    """Test creating an OpenAI response with multiple messages."""
    # Setup
    input_messages = [
        OpenAIResponseMessage(role="developer", content="You are a helpful assistant", name=None),
        OpenAIResponseMessage(role="user", content="Name some towns in Ireland", name=None),
        OpenAIResponseMessage(
            role="assistant",
            content=[
                OpenAIResponseInputMessageContentText(text="Galway, Longford, Sligo"),
                OpenAIResponseInputMessageContentText(text="Dublin"),
            ],
            name=None,
        ),
        OpenAIResponseMessage(role="user", content="Which is the largest town in Ireland?", name=None),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    await openai_responses_impl.create_openai_response(
        input=input_messages,
        model=model,
        temperature=0.1,
    )

    # Verify the the correct messages were sent to the inference API i.e.
    # All of the responses message were convered to the chat completion message objects
    call_args = mock_inference_api.openai_chat_completion.call_args_list[0]
    params = call_args.args[0]
    inference_messages = params.messages
    for i, m in enumerate(input_messages):
        if isinstance(m.content, str):
            assert inference_messages[i].content == m.content
        else:
            assert inference_messages[i].content[0].text == m.content[0].text
            assert isinstance(inference_messages[i].content[0], OpenAIChatCompletionContentPartTextParam)
        assert inference_messages[i].role == m.role
        if m.role == "user":
            assert isinstance(inference_messages[i], OpenAIUserMessageParam)
        elif m.role == "assistant":
            assert isinstance(inference_messages[i], OpenAIAssistantMessageParam)
        else:
            assert isinstance(inference_messages[i], OpenAIDeveloperMessageParam)


async def test_prepend_previous_response_basic(openai_responses_impl, mock_responses_store):
    """Test prepending a basic previous response to a new response."""

    input_item_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseInputMessageContentText(text="fake_previous_input")],
        role="user",
    )
    response_output_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseOutputMessageContentOutputText(text="fake_response")],
        status="completed",
        role="assistant",
    )
    previous_response = _OpenAIResponseObjectWithInputAndMessages(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[response_output_message],
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[input_item_message],
        messages=[OpenAIUserMessageParam(content="fake_previous_input")],
    )
    mock_responses_store.get_response_object.return_value = previous_response

    input = await openai_responses_impl._prepend_previous_response("fake_input", previous_response)

    assert len(input) == 3
    # Check for previous input
    assert isinstance(input[0], OpenAIResponseMessage)
    assert input[0].content[0].text == "fake_previous_input"
    # Check for previous output
    assert isinstance(input[1], OpenAIResponseMessage)
    assert input[1].content[0].text == "fake_response"
    # Check for new input
    assert isinstance(input[2], OpenAIResponseMessage)
    assert input[2].content == "fake_input"


async def test_prepend_previous_response_web_search(openai_responses_impl, mock_responses_store):
    """Test prepending a web search previous response to a new response."""
    input_item_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseInputMessageContentText(text="fake_previous_input")],
        role="user",
    )
    output_web_search = OpenAIResponseOutputMessageWebSearchToolCall(
        id="ws_123",
        status="completed",
    )
    output_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseOutputMessageContentOutputText(text="fake_web_search_response")],
        status="completed",
        role="assistant",
    )
    response = _OpenAIResponseObjectWithInputAndMessages(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[output_web_search, output_message],
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[input_item_message],
        messages=[OpenAIUserMessageParam(content="test input")],
    )
    mock_responses_store.get_response_object.return_value = response

    input_messages = [OpenAIResponseMessage(content="fake_input", role="user")]
    input = await openai_responses_impl._prepend_previous_response(input_messages, response)

    assert len(input) == 4
    # Check for previous input
    assert isinstance(input[0], OpenAIResponseMessage)
    assert input[0].content[0].text == "fake_previous_input"
    # Check for previous output web search tool call
    assert isinstance(input[1], OpenAIResponseOutputMessageWebSearchToolCall)
    # Check for previous output web search response
    assert isinstance(input[2], OpenAIResponseMessage)
    assert input[2].content[0].text == "fake_web_search_response"
    # Check for new input
    assert isinstance(input[3], OpenAIResponseMessage)
    assert input[3].content == "fake_input"


async def test_prepend_previous_response_mcp_tool_call(openai_responses_impl, mock_responses_store):
    """Test prepending a previous response which included an mcp tool call to a new response."""
    input_item_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseInputMessageContentText(text="fake_previous_input")],
        role="user",
    )
    output_tool_call = OpenAIResponseOutputMessageMCPCall(
        id="ws_123",
        name="fake-tool",
        arguments="fake-arguments",
        server_label="fake-label",
    )
    output_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseOutputMessageContentOutputText(text="fake_tool_call_response")],
        status="completed",
        role="assistant",
    )
    response = _OpenAIResponseObjectWithInputAndMessages(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[output_tool_call, output_message],
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[input_item_message],
        messages=[OpenAIUserMessageParam(content="test input")],
    )
    mock_responses_store.get_response_object.return_value = response

    input_messages = [OpenAIResponseMessage(content="fake_input", role="user")]
    input = await openai_responses_impl._prepend_previous_response(input_messages, response)

    assert len(input) == 4
    # Check for previous input
    assert isinstance(input[0], OpenAIResponseMessage)
    assert input[0].content[0].text == "fake_previous_input"
    # Check for previous output MCP tool call
    assert isinstance(input[1], OpenAIResponseOutputMessageMCPCall)
    # Check for previous output web search response
    assert isinstance(input[2], OpenAIResponseMessage)
    assert input[2].content[0].text == "fake_tool_call_response"
    # Check for new input
    assert isinstance(input[3], OpenAIResponseMessage)
    assert input[3].content == "fake_input"


async def test_create_openai_response_with_instructions(openai_responses_impl, mock_inference_api):
    # Setup
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    instructions = "You are a geography expert. Provide concise answers."

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        instructions=instructions,
    )

    # Verify
    mock_inference_api.openai_chat_completion.assert_called_once()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    sent_messages = params.messages

    # Check that instructions were prepended as a system message
    assert len(sent_messages) == 2
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == instructions
    assert sent_messages[1].role == "user"
    assert sent_messages[1].content == input_text


async def test_create_openai_response_with_instructions_and_multiple_messages(
    openai_responses_impl, mock_inference_api, mock_files_api
):
    # Setup
    input_messages = [
        OpenAIResponseMessage(role="user", content="Name some towns in Ireland", name=None),
        OpenAIResponseMessage(
            role="assistant",
            content="Galway, Longford, Sligo",
            name=None,
        ),
        OpenAIResponseMessage(role="user", content="Which is the largest?", name=None),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"
    instructions = "You are a geography expert. Provide concise answers."

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    await openai_responses_impl.create_openai_response(
        input=input_messages,
        model=model,
        instructions=instructions,
    )

    # Verify
    mock_inference_api.openai_chat_completion.assert_called_once()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    sent_messages = params.messages

    # Check that instructions were prepended as a system message
    assert len(sent_messages) == 4  # 1 system + 3 input messages
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == instructions

    # Check the rest of the messages were converted correctly
    assert sent_messages[1].role == "user"
    assert sent_messages[1].content == "Name some towns in Ireland"
    assert sent_messages[2].role == "assistant"
    assert sent_messages[2].content == "Galway, Longford, Sligo"
    assert sent_messages[3].role == "user"
    assert sent_messages[3].content == "Which is the largest?"


async def test_create_openai_response_with_instructions_and_previous_response(
    openai_responses_impl, mock_responses_store, mock_inference_api
):
    """Test prepending both instructions and previous response."""

    input_item_message = OpenAIResponseMessage(
        id="123",
        content="Name some towns in Ireland",
        role="user",
    )
    response_output_message = OpenAIResponseMessage(
        id="123",
        content="Galway, Longford, Sligo",
        status="completed",
        role="assistant",
    )
    response = _OpenAIResponseObjectWithInputAndMessages(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[response_output_message],
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[input_item_message],
        messages=[
            OpenAIUserMessageParam(content="Name some towns in Ireland"),
            OpenAIAssistantMessageParam(content="Galway, Longford, Sligo"),
        ],
    )
    mock_responses_store.get_response_object.return_value = response

    model = "meta-llama/Llama-3.1-8B-Instruct"
    instructions = "You are a geography expert. Provide concise answers."

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    await openai_responses_impl.create_openai_response(
        input="Which is the largest?", model=model, instructions=instructions, previous_response_id="123"
    )

    # Verify
    mock_inference_api.openai_chat_completion.assert_called_once()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    sent_messages = params.messages

    # Check that instructions were prepended as a system message
    assert len(sent_messages) == 4, sent_messages
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == instructions

    # Check the rest of the messages were converted correctly
    assert sent_messages[1].role == "user"
    assert sent_messages[1].content == "Name some towns in Ireland"
    assert sent_messages[2].role == "assistant"
    assert sent_messages[2].content == "Galway, Longford, Sligo"
    assert sent_messages[3].role == "user"
    assert sent_messages[3].content == "Which is the largest?"


async def test_create_openai_response_with_previous_response_instructions(
    openai_responses_impl, mock_responses_store, mock_inference_api
):
    """Test prepending instructions and previous response with instructions."""

    input_item_message = OpenAIResponseMessage(
        id="123",
        content="Name some towns in Ireland",
        role="user",
    )
    response_output_message = OpenAIResponseMessage(
        id="123",
        content="Galway, Longford, Sligo",
        status="completed",
        role="assistant",
    )
    response = _OpenAIResponseObjectWithInputAndMessages(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[response_output_message],
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[input_item_message],
        messages=[
            OpenAIUserMessageParam(content="Name some towns in Ireland"),
            OpenAIAssistantMessageParam(content="Galway, Longford, Sligo"),
        ],
        instructions="You are a helpful assistant.",
    )
    mock_responses_store.get_response_object.return_value = response

    model = "meta-llama/Llama-3.1-8B-Instruct"
    instructions = "You are a geography expert. Provide concise answers."

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    await openai_responses_impl.create_openai_response(
        input="Which is the largest?", model=model, instructions=instructions, previous_response_id="123"
    )

    # Verify
    mock_inference_api.openai_chat_completion.assert_called_once()
    call_args = mock_inference_api.openai_chat_completion.call_args
    params = call_args.args[0]
    sent_messages = params.messages

    # Check that instructions were prepended as a system message
    # and that the previous response instructions were not carried over
    assert len(sent_messages) == 4, sent_messages
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == instructions

    # Check the rest of the messages were converted correctly
    assert sent_messages[1].role == "user"
    assert sent_messages[1].content == "Name some towns in Ireland"
    assert sent_messages[2].role == "assistant"
    assert sent_messages[2].content == "Galway, Longford, Sligo"
    assert sent_messages[3].role == "user"
    assert sent_messages[3].content == "Which is the largest?"


async def test_list_openai_response_input_items_delegation(openai_responses_impl, mock_responses_store):
    """Test that list_openai_response_input_items properly delegates to responses_store with correct parameters."""
    # Setup
    response_id = "resp_123"
    after = "msg_after"
    before = "msg_before"
    include = ["metadata"]
    limit = 5
    order = Order.asc

    input_message = OpenAIResponseMessage(
        id="msg_123",
        content="Test message",
        role="user",
    )

    expected_result = ListOpenAIResponseInputItem(data=[input_message])
    mock_responses_store.list_response_input_items.return_value = expected_result

    # Execute with all parameters to test delegation
    result = await openai_responses_impl.list_openai_response_input_items(
        response_id, after=after, before=before, include=include, limit=limit, order=order
    )

    # Verify all parameters are passed through correctly to the store
    mock_responses_store.list_response_input_items.assert_called_once_with(
        response_id, after, before, include, limit, order
    )

    # Verify the result is returned as-is from the store
    assert result.object == "list"
    assert len(result.data) == 1
    assert result.data[0].id == "msg_123"


async def test_responses_store_list_input_items_logic():
    """Test ResponsesStore list_response_input_items logic - mocks get_response_object to test actual ordering/limiting."""

    # Create mock store and response store
    mock_sql_store = AsyncMock()
    backend_name = "sql_responses_test"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path="mock_db_path")})
    responses_store = ResponsesStore(
        ResponsesStoreReference(backend=backend_name, table_name="responses"), policy=default_policy()
    )
    responses_store.sql_store = mock_sql_store

    # Setup test data - multiple input items
    input_items = [
        OpenAIResponseMessage(id="msg_1", content="First message", role="user"),
        OpenAIResponseMessage(id="msg_2", content="Second message", role="user"),
        OpenAIResponseMessage(id="msg_3", content="Third message", role="user"),
        OpenAIResponseMessage(id="msg_4", content="Fourth message", role="user"),
    ]

    response_with_input = _OpenAIResponseObjectWithInputAndMessages(
        id="resp_123",
        model="test_model",
        created_at=1234567890,
        object="response",
        status="completed",
        output=[],
        text=OpenAIResponseText(format=(OpenAIResponseTextFormat(type="text"))),
        input=input_items,
        messages=[OpenAIUserMessageParam(content="First message")],
    )

    # Mock the get_response_object method to return our test data
    mock_sql_store.fetch_one.return_value = {"response_object": response_with_input.model_dump()}

    # Test 1: Default behavior (no limit, desc order)
    result = await responses_store.list_response_input_items("resp_123")
    assert result.object == "list"
    assert len(result.data) == 4
    # Should be reversed for desc order
    assert result.data[0].id == "msg_4"
    assert result.data[1].id == "msg_3"
    assert result.data[2].id == "msg_2"
    assert result.data[3].id == "msg_1"

    # Test 2: With limit=2, desc order
    result = await responses_store.list_response_input_items("resp_123", limit=2, order=Order.desc)
    assert result.object == "list"
    assert len(result.data) == 2
    # Should be first 2 items in desc order
    assert result.data[0].id == "msg_4"
    assert result.data[1].id == "msg_3"

    # Test 3: With limit=2, asc order
    result = await responses_store.list_response_input_items("resp_123", limit=2, order=Order.asc)
    assert result.object == "list"
    assert len(result.data) == 2
    # Should be first 2 items in original order (asc)
    assert result.data[0].id == "msg_1"
    assert result.data[1].id == "msg_2"

    # Test 4: Asc order without limit
    result = await responses_store.list_response_input_items("resp_123", order=Order.asc)
    assert result.object == "list"
    assert len(result.data) == 4
    # Should be in original order (asc)
    assert result.data[0].id == "msg_1"
    assert result.data[1].id == "msg_2"
    assert result.data[2].id == "msg_3"
    assert result.data[3].id == "msg_4"

    # Test 5: Large limit (larger than available items)
    result = await responses_store.list_response_input_items("resp_123", limit=10, order=Order.desc)
    assert result.object == "list"
    assert len(result.data) == 4  # Should return all available items
    assert result.data[0].id == "msg_4"

    # Test 6: Zero limit edge case
    result = await responses_store.list_response_input_items("resp_123", limit=0, order=Order.asc)
    assert result.object == "list"
    assert len(result.data) == 0  # Should return no items


async def test_store_response_uses_rehydrated_input_with_previous_response(
    openai_responses_impl, mock_responses_store, mock_inference_api
):
    """Test that _store_response uses the full re-hydrated input (including previous responses)
    rather than just the original input when previous_response_id is provided."""

    # Setup - Create a previous response that should be included in the stored input
    previous_response = _OpenAIResponseObjectWithInputAndMessages(
        id="resp-previous-123",
        object="response",
        created_at=1234567890,
        model="meta-llama/Llama-3.1-8B-Instruct",
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[
            OpenAIResponseMessage(
                id="msg-prev-user", role="user", content=[OpenAIResponseInputMessageContentText(text="What is 2+2?")]
            )
        ],
        output=[
            OpenAIResponseMessage(
                id="msg-prev-assistant",
                role="assistant",
                content=[OpenAIResponseOutputMessageContentOutputText(text="2+2 equals 4.")],
            )
        ],
        messages=[
            OpenAIUserMessageParam(content="What is 2+2?"),
            OpenAIAssistantMessageParam(content="2+2 equals 4."),
        ],
    )

    mock_responses_store.get_response_object.return_value = previous_response

    current_input = "Now what is 3+3?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute - Create response with previous_response_id
    result = await openai_responses_impl.create_openai_response(
        input=current_input,
        model=model,
        previous_response_id="resp-previous-123",
        store=True,
    )

    store_call_args = mock_responses_store.store_response_object.call_args
    stored_input = store_call_args.kwargs["input"]

    # Verify that the stored input contains the full re-hydrated conversation:
    # 1. Previous user message
    # 2. Previous assistant response
    # 3. Current user message
    assert len(stored_input) == 3

    assert stored_input[0].role == "user"
    assert stored_input[0].content[0].text == "What is 2+2?"

    assert stored_input[1].role == "assistant"
    assert stored_input[1].content[0].text == "2+2 equals 4."

    assert stored_input[2].role == "user"
    assert stored_input[2].content == "Now what is 3+3?"

    # Verify the response itself is correct
    assert result.model == model
    assert result.status == "completed"


@patch("llama_stack.providers.utils.tools.mcp.list_mcp_tools")
async def test_reuse_mcp_tool_list(
    mock_list_mcp_tools, openai_responses_impl, mock_responses_store, mock_inference_api
):
    """Test that mcp_list_tools can be reused where appropriate."""

    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    mock_list_mcp_tools.return_value = ListToolDefsResponse(
        data=[ToolDef(name="test_tool", description="a test tool", input_schema={}, output_schema={})]
    )

    res1 = await openai_responses_impl.create_openai_response(
        input="What is 2+2?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        store=True,
        tools=[
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ],
    )
    args = mock_responses_store.store_response_object.call_args
    data = args.kwargs["response_object"].model_dump()
    data["input"] = [input_item.model_dump() for input_item in args.kwargs["input"]]
    data["messages"] = [msg.model_dump() for msg in args.kwargs["messages"]]
    stored = _OpenAIResponseObjectWithInputAndMessages(**data)
    mock_responses_store.get_response_object.return_value = stored

    res2 = await openai_responses_impl.create_openai_response(
        previous_response_id=res1.id,
        input="Now what is 3+3?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        store=True,
        tools=[
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ],
    )
    assert len(mock_inference_api.openai_chat_completion.call_args_list) == 2
    second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
    second_params = second_call.args[0]
    tools_seen = second_params.tools
    assert len(tools_seen) == 1
    assert tools_seen[0]["function"]["name"] == "test_tool"
    assert tools_seen[0]["function"]["description"] == "a test tool"

    assert mock_list_mcp_tools.call_count == 1
    listings = [obj for obj in res2.output if obj.type == "mcp_list_tools"]
    assert len(listings) == 1
    assert listings[0].server_label == "alabel"
    assert len(listings[0].tools) == 1
    assert listings[0].tools[0].name == "test_tool"


@pytest.mark.parametrize(
    "text_format, response_format",
    [
        (OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")), None),
        (
            OpenAIResponseText(format=OpenAIResponseTextFormat(name="Test", schema={"foo": "bar"}, type="json_schema")),
            OpenAIResponseFormatJSONSchema(json_schema=OpenAIJSONSchema(name="Test", schema={"foo": "bar"})),
        ),
        (OpenAIResponseText(format=OpenAIResponseTextFormat(type="json_object")), OpenAIResponseFormatJSONObject()),
        # ensure text param with no format specified defaults to None
        (OpenAIResponseText(format=None), None),
        # ensure text param of None defaults to None
        (None, None),
    ],
)
async def test_create_openai_response_with_text_format(
    openai_responses_impl, mock_inference_api, text_format, response_format
):
    """Test creating Responses with text formats."""
    # Setup
    input_text = "How hot it is in San Francisco today?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    _result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        text=text_format,
    )

    # Verify
    first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
    first_params = first_call.args[0]
    assert first_params.messages[0].content == input_text
    assert first_params.response_format == response_format


async def test_create_openai_response_with_invalid_text_format(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with an invalid text format."""
    # Setup
    input_text = "How hot it is in San Francisco today?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Execute
    with pytest.raises(ValueError):
        _result = await openai_responses_impl.create_openai_response(
            input=input_text,
            model=model,
            text=OpenAIResponseText(format={"type": "invalid"}),
        )


async def test_create_openai_response_with_prompt(openai_responses_impl, mock_inference_api, mock_prompts_api):
    """Test creating an OpenAI response with a prompt."""
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="You are a helpful {{ area_name }} assistant at {{ company_name }}. Always provide accurate information.",
        prompt_id=prompt_id,
        version=1,
        variables=["area_name", "company_name"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentText,
        OpenAIResponsePromptParam,
    )

    prompt_params_with_version_1 = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "area_name": OpenAIResponseInputMessageContentText(text="geography"),
            "company_name": OpenAIResponseInputMessageContentText(text="Dummy Company"),
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        prompt=prompt_params_with_version_1,
    )

    mock_prompts_api.get_prompt.assert_called_with(prompt_id, 1)
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    sent_messages = call_args.args[0].messages
    assert len(sent_messages) == 2

    system_messages = [msg for msg in sent_messages if msg.role == "system"]
    assert len(system_messages) == 1
    assert (
        system_messages[0].content
        == "You are a helpful geography assistant at Dummy Company. Always provide accurate information."
    )

    user_messages = [msg for msg in sent_messages if msg.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].content == input_text

    assert result.model == model
    assert result.status == "completed"
    assert result.prompt.prompt_id == prompt_id
    assert result.prompt.variables == ["area_name", "company_name"]
    assert result.prompt.version == 1
    assert result.prompt.prompt == prompt.prompt


async def test_prepend_prompt_successful_without_variables(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt function without variables."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="You are a helpful assistant. Always provide accurate information.",
        prompt_id=prompt_id,
        version=1,
        variables=[],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import OpenAIResponsePromptParam
    from llama_stack.apis.inference import OpenAISystemMessageParam, OpenAIUserMessageParam

    prompt_params = OpenAIResponsePromptParam(id=prompt_id, version="1")

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Hello")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    mock_prompts_api.get_prompt.assert_called_once_with(prompt_id, 1)

    # Check that prompt was returned
    assert result == prompt

    # Check that system message was prepended
    assert len(messages) == 2
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "You are a helpful assistant. Always provide accurate information."


async def test_prepend_prompt_no_version_specified(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt function when no version is specified (should use None)."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Default prompt text.",
        prompt_id=prompt_id,
        version=3,
        variables=[],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import OpenAIResponsePromptParam
    from llama_stack.apis.inference import OpenAIUserMessageParam

    prompt_params = OpenAIResponsePromptParam(id=prompt_id)  # No version specified

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    mock_prompts_api.get_prompt.assert_called_once_with(prompt_id, None)
    assert result == prompt
    assert len(messages) == 2


async def test_prepend_prompt_invalid_variable(openai_responses_impl, mock_prompts_api):
    """Test error handling in prepend_prompt function when prompt parameters contain invalid variables."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="You are a {{ role }} assistant.",
        prompt_id=prompt_id,
        version=1,
        variables=["role"],  # Only "role" is valid
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentText,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.inference import OpenAIUserMessageParam

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "role": OpenAIResponseInputMessageContentText(text="helpful"),
            "company": OpenAIResponseInputMessageContentText(
                text="Dummy Company"
            ),  # company is not in prompt.variables
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test prompt")]

    # Execute - should raise ValueError for invalid variable
    with pytest.raises(ValueError, match="Variable company not found in prompt"):
        await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    mock_prompts_api.get_prompt.assert_called_once_with(prompt_id, 1)


async def test_prepend_prompt_not_found(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt function when prompt is not found."""
    # Setup
    prompt_id = "pmpt_nonexistent"

    from llama_stack.apis.agents.openai_responses import OpenAIResponsePromptParam
    from llama_stack.apis.inference import OpenAIUserMessageParam

    prompt_params = OpenAIResponsePromptParam(id=prompt_id, version="1")

    mock_prompts_api.get_prompt.return_value = None  # Prompt not found

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test prompt")]
    initial_length = len(messages)

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    mock_prompts_api.get_prompt.assert_called_once_with(prompt_id, 1)

    # Should return None when prompt not found
    assert result is None

    # Messages should not be modified
    assert len(messages) == initial_length
    assert messages[0].content == "Test prompt"


async def test_prepend_prompt_no_params(openai_responses_impl, mock_prompts_api):
    """Test handling in prepend_prompt function when prompt_params is None."""
    # Setup
    from llama_stack.apis.inference import OpenAIUserMessageParam

    messages = [OpenAIUserMessageParam(content="Test")]
    initial_length = len(messages)

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, None)

    # Verify
    mock_prompts_api.get_prompt.assert_not_called()

    # Should return None when no prompt params
    assert result is None

    # Messages should not be modified
    assert len(messages) == initial_length


async def test_prepend_prompt_variable_substitution(openai_responses_impl, mock_prompts_api):
    """Test complex variable substitution with multiple occurrences and special characters in prepend_prompt function."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"

    # Support all whitespace variations: {{name}}, {{ name }}, {{ name}}, {{name }}, etc.
    prompt = Prompt(
        prompt="Hello {{name}}! You are working at {{ company}}. Your role is {{role}} at {{company}}. Remember, {{ name }}, to be {{ tone }}.",
        prompt_id=prompt_id,
        version=1,
        variables=["name", "company", "role", "tone"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentText,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.inference import OpenAISystemMessageParam, OpenAIUserMessageParam

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "name": OpenAIResponseInputMessageContentText(text="Alice"),
            "company": OpenAIResponseInputMessageContentText(text="Dummy Company"),
            "role": OpenAIResponseInputMessageContentText(text="AI Assistant"),
            "tone": OpenAIResponseInputMessageContentText(text="professional"),
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    assert result == prompt
    assert len(messages) == 2
    assert isinstance(messages[0], OpenAISystemMessageParam)
    expected_content = "Hello Alice! You are working at Dummy Company. Your role is AI Assistant at Dummy Company. Remember, Alice, to be professional."
    assert messages[0].content == expected_content


async def test_prepend_prompt_with_image_variable(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with image variable - should create placeholder in system message and inject image into user message."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Analyze this {{product_image}} and describe what you see.",
        prompt_id=prompt_id,
        version=1,
        variables=["product_image"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentImage,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.inference import (
        OpenAIChatCompletionContentPartImageParam,
        OpenAISystemMessageParam,
        OpenAIUserMessageParam,
    )

    # Mock file content
    mock_file_content = b"fake_image_data"
    mock_files_api.openai_retrieve_file_content.return_value = type("obj", (object,), {"body": mock_file_content})()

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "product_image": OpenAIResponseInputMessageContentImage(
                file_id="file-abc123",
                detail="high",
            )
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="What do you think?")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    assert result == prompt
    assert len(messages) == 2

    # Check system message has placeholder
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Analyze this [Image: product_image] and describe what you see."

    # Check user message has image prepended
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert isinstance(messages[1].content, list)
    assert len(messages[1].content) == 2  # Image + original text

    # First part should be image with data URL
    assert isinstance(messages[1].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[1].content[0].image_url.url.startswith("data:image/")
    assert messages[1].content[0].image_url.detail == "high"

    # Second part should be original text
    assert messages[1].content[1].text == "What do you think?"


async def test_prepend_prompt_with_file_variable(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with file variable - should create placeholder in system message and inject file into user message."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Review the document {{contract_file}} and summarize key points.",
        prompt_id=prompt_id,
        version=1,
        variables=["contract_file"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentFile,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.files import OpenAIFileObject
    from llama_stack.apis.inference import (
        OpenAIFile,
        OpenAISystemMessageParam,
        OpenAIUserMessageParam,
    )

    # Mock file retrieval
    mock_file_content = b"fake_pdf_content"
    mock_files_api.openai_retrieve_file_content.return_value = type("obj", (object,), {"body": mock_file_content})()
    mock_files_api.openai_retrieve_file.return_value = OpenAIFileObject(
        object="file",
        id="file-contract-789",
        bytes=len(mock_file_content),
        created_at=1234567890,
        expires_at=1234567890,
        filename="contract.pdf",
        purpose="assistants",
    )

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "contract_file": OpenAIResponseInputMessageContentFile(
                file_id="file-contract-789",
                filename="contract.pdf",
            )
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Please review this.")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    assert result == prompt
    assert len(messages) == 2

    # Check system message has placeholder
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Review the document [File: contract_file] and summarize key points."

    # Check user message has file prepended
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert isinstance(messages[1].content, list)
    assert len(messages[1].content) == 2  # File + original text

    # First part should be file with data URL (not file_id)
    assert isinstance(messages[1].content[0], OpenAIFile)
    assert messages[1].content[0].file.file_data.startswith("data:application/pdf;base64,")
    assert messages[1].content[0].file.filename == "contract.pdf"
    # file_id should NOT be set in the OpenAI request
    assert messages[1].content[0].file.file_id is None

    # Second part should be original text
    assert messages[1].content[1].text == "Please review this."


async def test_prepend_prompt_with_mixed_variables(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with text, image, and file variables mixed together."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Hello {{name}}! Analyze {{photo}} and review {{document}}. Provide insights for {{company}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["name", "photo", "document", "company"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentFile,
        OpenAIResponseInputMessageContentImage,
        OpenAIResponseInputMessageContentText,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.files import OpenAIFileObject
    from llama_stack.apis.inference import (
        OpenAIChatCompletionContentPartImageParam,
        OpenAIFile,
        OpenAISystemMessageParam,
        OpenAIUserMessageParam,
    )

    # Mock file retrieval for document
    mock_file_content = b"fake_doc_content"
    mock_files_api.openai_retrieve_file_content.return_value = type("obj", (object,), {"body": mock_file_content})()
    mock_files_api.openai_retrieve_file.return_value = OpenAIFileObject(
        object="file",
        id="file-doc-456",
        bytes=len(mock_file_content),
        created_at=1234567890,
        expires_at=1234567890,
        filename="doc.pdf",
        purpose="assistants",
    )

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "name": OpenAIResponseInputMessageContentText(text="Alice"),
            "photo": OpenAIResponseInputMessageContentImage(file_id="file-photo-123", detail="auto"),
            "document": OpenAIResponseInputMessageContentFile(file_id="file-doc-456", filename="doc.pdf"),
            "company": OpenAIResponseInputMessageContentText(text="Acme Corp"),
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Here's my question.")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    assert result == prompt
    assert len(messages) == 2

    # Check system message has text and placeholders
    assert isinstance(messages[0], OpenAISystemMessageParam)
    expected_system = "Hello Alice! Analyze [Image: photo] and review [File: document]. Provide insights for Acme Corp."
    assert messages[0].content == expected_system

    # Check user message has media prepended (2 media items + original text)
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert isinstance(messages[1].content, list)
    assert len(messages[1].content) == 3  # Image + File + original text

    # First part should be image with data URL
    assert isinstance(messages[1].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[1].content[0].image_url.url.startswith("data:image/")

    # Second part should be file with data URL
    assert isinstance(messages[1].content[1], OpenAIFile)
    assert messages[1].content[1].file.file_data.startswith("data:application/pdf;base64,")
    assert messages[1].content[1].file.filename == "doc.pdf"
    assert messages[1].content[1].file.file_id is None  # file_id should NOT be sent

    # Third part should be original text
    assert messages[1].content[2].text == "Here's my question."


async def test_prepend_prompt_with_image_using_image_url(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt with image variable using image_url instead of file_id."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Describe {{screenshot}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["screenshot"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentImage,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.inference import (
        OpenAIChatCompletionContentPartImageParam,
        OpenAISystemMessageParam,
        OpenAIUserMessageParam,
    )

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={
            "screenshot": OpenAIResponseInputMessageContentImage(
                image_url="https://example.com/screenshot.png",
                detail="low",
            )
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="What is this?")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    assert result == prompt
    assert len(messages) == 2

    # Check system message has placeholder
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Describe [Image: screenshot]."

    # Check user message has image with URL
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert isinstance(messages[1].content, list)

    # Image should use the provided URL
    assert isinstance(messages[1].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[1].content[0].image_url.url == "https://example.com/screenshot.png"
    assert messages[1].content[0].image_url.detail == "low"


async def test_prepend_prompt_with_media_no_user_message(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with media when there's no existing user message - should create one."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Analyze {{image}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["image"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentImage,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.inference import (
        OpenAIAssistantMessageParam,
        OpenAIChatCompletionContentPartImageParam,
        OpenAISystemMessageParam,
        OpenAIUserMessageParam,
    )

    # Mock file content
    mock_file_content = b"fake_image_data"
    mock_files_api.openai_retrieve_file_content.return_value = type("obj", (object,), {"body": mock_file_content})()

    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={"image": OpenAIResponseInputMessageContentImage(file_id="file-img-999")},
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages - only assistant message, no user message
    messages = [OpenAIAssistantMessageParam(content="Previous response")]

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, prompt_params)

    # Verify
    assert result == prompt
    assert len(messages) == 3  # System + Assistant + New User

    # Check system message
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Analyze [Image: image]."

    # Original assistant message should still be there
    assert isinstance(messages[1], OpenAIAssistantMessageParam)
    assert messages[1].content == "Previous response"

    # New user message with just the image should be appended
    assert isinstance(messages[2], OpenAIUserMessageParam)
    assert isinstance(messages[2].content, list)
    assert len(messages[2].content) == 1
    assert isinstance(messages[2].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[2].content[0].image_url.url.startswith("data:image/")


async def test_prepend_prompt_image_variable_missing_required_fields(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt with image variable that has neither file_id nor image_url - should raise error."""
    # Setup
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Analyze {{bad_image}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["bad_image"],
        is_default=True,
    )

    from llama_stack.apis.agents.openai_responses import (
        OpenAIResponseInputMessageContentImage,
        OpenAIResponsePromptParam,
    )
    from llama_stack.apis.inference import OpenAIUserMessageParam

    # Create image content with neither file_id nor image_url
    prompt_params = OpenAIResponsePromptParam(
        id=prompt_id,
        version="1",
        variables={"bad_image": OpenAIResponseInputMessageContentImage()},  # No file_id or image_url
    )

    mock_prompts_api.get_prompt.return_value = prompt
    messages = [OpenAIUserMessageParam(content="Test")]

    # Execute - should raise ValueError
    with pytest.raises(ValueError, match="Image content must have either 'image_url' or 'file_id'"):
        await openai_responses_impl._prepend_prompt(messages, prompt_params)
