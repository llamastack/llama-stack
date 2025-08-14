# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import time

import httpx
import openai
import pytest

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.core.datatypes import AuthenticationRequiredError
from tests.common.mcp import dependency_tools, make_mcp_server

from .fixtures.fixtures import case_id_generator
from .fixtures.load import load_test_cases

responses_test_cases = load_test_cases("responses")


def _new_vector_store(openai_client, name):
    # Ensure we don't reuse an existing vector store
    vector_stores = openai_client.vector_stores.list()
    for vector_store in vector_stores:
        if vector_store.name == name:
            openai_client.vector_stores.delete(vector_store_id=vector_store.id)

    # Create a new vector store
    vector_store = openai_client.vector_stores.create(
        name=name,
    )
    return vector_store


def _upload_file(openai_client, name, file_path):
    # Ensure we don't reuse an existing file
    files = openai_client.files.list()
    for file in files:
        if file.filename == name:
            openai_client.files.delete(file_id=file.id)

    # Upload a text file with our document content
    return openai_client.files.create(file=open(file_path, "rb"), purpose="assistants")


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_basic(request, compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        stream=False,
    )
    output_text = response.output_text.lower().strip()
    assert len(output_text) > 0
    assert case["output"].lower() in output_text

    retrieved_response = compat_client.responses.retrieve(response_id=response.id)
    assert retrieved_response.output_text == response.output_text

    next_response = compat_client.responses.create(
        model=text_model_id,
        input="Repeat your previous response in all caps.",
        previous_response_id=response.id,
    )
    next_output_text = next_response.output_text.strip()
    assert case["output"].upper() in next_output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_streaming_basic(request, compat_client, text_model_id, case):
    import time

    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        stream=True,
    )

    # Track events and timing to verify proper streaming
    events = []
    event_times = []
    response_id = ""

    start_time = time.time()

    for chunk in response:
        current_time = time.time()
        event_times.append(current_time - start_time)
        events.append(chunk)

        if chunk.type == "response.created":
            # Verify response.created is emitted first and immediately
            assert len(events) == 1, "response.created should be the first event"
            assert event_times[0] < 0.1, "response.created should be emitted immediately"
            assert chunk.response.status == "in_progress"
            response_id = chunk.response.id

        elif chunk.type == "response.completed":
            # Verify response.completed comes after response.created
            assert len(events) >= 2, "response.completed should come after response.created"
            assert chunk.response.status == "completed"
            assert chunk.response.id == response_id, "Response ID should be consistent"

            # Verify content quality
            output_text = chunk.response.output_text.lower().strip()
            assert len(output_text) > 0, "Response should have content"
            assert case["output"].lower() in output_text, f"Expected '{case['output']}' in response"

    # Verify we got both required events
    event_types = [event.type for event in events]
    assert "response.created" in event_types, "Missing response.created event"
    assert "response.completed" in event_types, "Missing response.completed event"

    # Verify event order
    created_index = event_types.index("response.created")
    completed_index = event_types.index("response.completed")
    assert created_index < completed_index, "response.created should come before response.completed"

    # Verify stored response matches streamed response
    retrieved_response = compat_client.responses.retrieve(response_id=response_id)
    final_event = events[-1]
    assert retrieved_response.output_text == final_event.response.output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_streaming_incremental_content(request, compat_client, text_model_id, case):
    """Test that streaming actually delivers content incrementally, not just at the end."""
    import time

    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        stream=True,
    )

    # Track all events and their content to verify incremental streaming
    events = []
    content_snapshots = []
    event_times = []

    start_time = time.time()

    for chunk in response:
        current_time = time.time()
        event_times.append(current_time - start_time)
        events.append(chunk)

        # Track content at each event based on event type
        if chunk.type == "response.output_text.delta":
            # For delta events, track the delta content
            content_snapshots.append(chunk.delta)
        elif hasattr(chunk, "response") and hasattr(chunk.response, "output_text"):
            # For response.created/completed events, track the full output_text
            content_snapshots.append(chunk.response.output_text)
        else:
            content_snapshots.append("")

    # Verify we have the expected events
    event_types = [event.type for event in events]
    assert "response.created" in event_types, "Missing response.created event"
    assert "response.completed" in event_types, "Missing response.completed event"

    # Check if we have incremental content updates
    created_index = event_types.index("response.created")
    completed_index = event_types.index("response.completed")

    # The key test: verify content progression
    created_content = content_snapshots[created_index]
    completed_content = content_snapshots[completed_index]

    # Verify that response.created has empty or minimal content
    assert len(created_content) == 0, f"response.created should have empty content, got: {repr(created_content[:100])}"

    # Verify that response.completed has the full content
    assert len(completed_content) > 0, "response.completed should have content"
    assert case["output"].lower() in completed_content.lower(), f"Expected '{case['output']}' in final content"

    # Check for true incremental streaming by looking for delta events
    delta_events = [i for i, event_type in enumerate(event_types) if event_type == "response.output_text.delta"]

    # Assert that we have delta events (true incremental streaming)
    assert len(delta_events) > 0, "Expected delta events for true incremental streaming, but found none"

    # Verify delta events have content and accumulate to final content
    delta_content_total = ""
    non_empty_deltas = 0

    for delta_idx in delta_events:
        delta_content = content_snapshots[delta_idx]
        if delta_content:
            delta_content_total += delta_content
            non_empty_deltas += 1

    # Assert that we have meaningful delta content
    assert non_empty_deltas > 0, "Delta events found but none contain content"
    assert len(delta_content_total) > 0, "Delta events found but total delta content is empty"

    # Verify that the accumulated delta content matches the final content
    assert delta_content_total.strip() == completed_content.strip(), (
        f"Delta content '{delta_content_total}' should match final content '{completed_content}'"
    )

    # Verify timing: delta events should come between created and completed
    for delta_idx in delta_events:
        assert created_index < delta_idx < completed_index, (
            f"Delta event at index {delta_idx} should be between created ({created_index}) and completed ({completed_index})"
        )


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_multi_turn"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_multi_turn(request, compat_client, text_model_id, case):
    previous_response_id = None
    for turn in case["turns"]:
        response = compat_client.responses.create(
            model=text_model_id,
            input=turn["input"],
            previous_response_id=previous_response_id,
            tools=turn["tools"] if "tools" in turn else None,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn["output"].lower() in output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_web_search"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_web_search(request, compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        tools=case["tools"],
        stream=False,
    )
    assert len(response.output) > 1
    assert response.output[0].type == "web_search_call"
    assert response.output[0].status == "completed"
    assert response.output[1].type == "message"
    assert response.output[1].status == "completed"
    assert response.output[1].role == "assistant"
    assert len(response.output[1].content) > 0
    assert case["output"].lower() in response.output_text.lower().strip()


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_file_search"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_file_search(request, compat_client, text_model_id, tmp_path, case):
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("Responses API file search is not yet supported in library client.")

    vector_store = _new_vector_store(compat_client, "test_vector_store")

    if "file_content" in case:
        file_name = "test_response_non_streaming_file_search.txt"
        file_path = tmp_path / file_name
        file_path.write_text(case["file_content"])
    elif "file_path" in case:
        file_path = os.path.join(os.path.dirname(__file__), "fixtures", case["file_path"])
        file_name = os.path.basename(file_path)
    else:
        raise ValueError(f"No file content or path provided for case {case['case_id']}")

    file_response = _upload_file(compat_client, file_name, file_path)

    # Attach our file to the vector store
    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_response.id,
    )

    # Wait for the file to be attached
    while file_attach_response.status == "in_progress":
        time.sleep(0.1)
        file_attach_response = compat_client.vector_stores.files.retrieve(
            vector_store_id=vector_store.id,
            file_id=file_response.id,
        )
    assert file_attach_response.status == "completed", f"Expected file to be attached, got {file_attach_response}"
    assert not file_attach_response.last_error

    # Update our tools with the right vector store id
    tools = case["tools"]
    for tool in tools:
        if tool["type"] == "file_search":
            tool["vector_store_ids"] = [vector_store.id]

    # Create the response request, which should query our vector store
    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    # Verify the file_search_tool was called
    assert len(response.output) > 1
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].queries  # ensure it's some non-empty list
    assert response.output[0].results
    assert case["output"].lower() in response.output[0].results[0].text.lower()
    assert response.output[0].results[0].score > 0

    # Verify the output_text generated by the response
    assert case["output"].lower() in response.output_text.lower().strip()


def test_response_non_streaming_file_search_empty_vector_store(request, compat_client, text_model_id):
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("Responses API file search is not yet supported in library client.")

    vector_store = _new_vector_store(compat_client, "test_vector_store")

    # Create the response request, which should query our vector store
    response = compat_client.responses.create(
        model=text_model_id,
        input="How many experts does the Llama 4 Maverick model have?",
        tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
        stream=False,
        include=["file_search_call.results"],
    )

    # Verify the file_search_tool was called
    assert len(response.output) > 1
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].queries  # ensure it's some non-empty list
    assert not response.output[0].results  # ensure we don't get any results

    # Verify some output_text was generated by the response
    assert response.output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_mcp_tool"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_mcp_tool(request, compat_client, text_model_id, case):
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    with make_mcp_server() as mcp_server_info:
        tools = case["tools"]
        for tool in tools:
            if tool["type"] == "mcp":
                tool["server_url"] = mcp_server_info["server_url"]

        response = compat_client.responses.create(
            model=text_model_id,
            input=case["input"],
            tools=tools,
            stream=False,
        )

        assert len(response.output) >= 3
        list_tools = response.output[0]
        assert list_tools.type == "mcp_list_tools"
        assert list_tools.server_label == "localmcp"
        assert len(list_tools.tools) == 2
        assert {t.name for t in list_tools.tools} == {
            "get_boiling_point",
            "greet_everyone",
        }

        call = response.output[1]
        assert call.type == "mcp_call"
        assert call.name == "get_boiling_point"
        assert json.loads(call.arguments) == {
            "liquid_name": "myawesomeliquid",
            "celsius": True,
        }
        assert call.error is None
        assert "-100" in call.output

        # sometimes the model will call the tool again, so we need to get the last message
        message = response.output[-1]
        text_content = message.content[0].text
        assert "boiling point" in text_content.lower()

    with make_mcp_server(required_auth_token="test-token") as mcp_server_info:
        tools = case["tools"]
        for tool in tools:
            if tool["type"] == "mcp":
                tool["server_url"] = mcp_server_info["server_url"]

        exc_type = (
            AuthenticationRequiredError
            if isinstance(compat_client, LlamaStackAsLibraryClient)
            else (httpx.HTTPStatusError, openai.AuthenticationError)
        )
        with pytest.raises(exc_type):
            compat_client.responses.create(
                model=text_model_id,
                input=case["input"],
                tools=tools,
                stream=False,
            )

        for tool in tools:
            if tool["type"] == "mcp":
                tool["server_url"] = mcp_server_info["server_url"]
                tool["headers"] = {"Authorization": "Bearer test-token"}

        response = compat_client.responses.create(
            model=text_model_id,
            input=case["input"],
            tools=tools,
            stream=False,
        )
        assert len(response.output) >= 3


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_custom_tool"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_custom_tool(request, compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        tools=case["tools"],
        stream=False,
    )
    assert len(response.output) == 1
    assert response.output[0].type == "function_call"
    assert response.output[0].status == "completed"
    assert response.output[0].name == "get_weather"


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_image"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_image(request, compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case["input"],
        stream=False,
    )
    output_text = response.output_text.lower()
    assert case["output"].lower() in output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_multi_turn_image"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_multi_turn_image(request, compat_client, text_model_id, case):
    previous_response_id = None
    for turn in case["turns"]:
        response = compat_client.responses.create(
            model=text_model_id,
            input=turn["input"],
            previous_response_id=previous_response_id,
            tools=turn["tools"] if "tools" in turn else None,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn["output"].lower() in output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_multi_turn_tool_execution"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_multi_turn_tool_execution(compat_client, text_model_id, case):
    """Test multi-turn tool execution where multiple MCP tool calls are performed in sequence."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    with make_mcp_server(tools=dependency_tools()) as mcp_server_info:
        tools = case["tools"]
        # Replace the placeholder URL with the actual server URL
        for tool in tools:
            if tool["type"] == "mcp" and tool["server_url"] == "<FILLED_BY_TEST_RUNNER>":
                tool["server_url"] = mcp_server_info["server_url"]

        response = compat_client.responses.create(
            input=case["input"],
            model=text_model_id,
            tools=tools,
        )

        # Verify we have MCP tool calls in the output
        mcp_list_tools = [output for output in response.output if output.type == "mcp_list_tools"]

        mcp_calls = [output for output in response.output if output.type == "mcp_call"]
        message_outputs = [output for output in response.output if output.type == "message"]

        # Should have exactly 1 MCP list tools message (at the beginning)
        assert len(mcp_list_tools) == 1, f"Expected exactly 1 mcp_list_tools, got {len(mcp_list_tools)}"
        assert mcp_list_tools[0].server_label == "localmcp"
        assert len(mcp_list_tools[0].tools) == 5  # Updated for dependency tools
        expected_tool_names = {
            "get_user_id",
            "get_user_permissions",
            "check_file_access",
            "get_experiment_id",
            "get_experiment_results",
        }
        assert {t.name for t in mcp_list_tools[0].tools} == expected_tool_names

        assert len(mcp_calls) >= 1, f"Expected at least 1 mcp_call, got {len(mcp_calls)}"
        for mcp_call in mcp_calls:
            assert mcp_call.error is None, f"MCP call should not have errors, got: {mcp_call.error}"

        assert len(message_outputs) >= 1, f"Expected at least 1 message output, got {len(message_outputs)}"

        final_message = message_outputs[-1]
        assert final_message.role == "assistant", f"Final message should be from assistant, got {final_message.role}"
        assert final_message.status == "completed", f"Final message should be completed, got {final_message.status}"
        assert len(final_message.content) > 0, "Final message should have content"

        expected_output = case["output"]
        assert expected_output.lower() in response.output_text.lower(), (
            f"Expected '{expected_output}' to appear in response: {response.output_text}"
        )


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_multi_turn_tool_execution_streaming"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_streaming_multi_turn_tool_execution(compat_client, text_model_id, case):
    """Test streaming multi-turn tool execution where multiple MCP tool calls are performed in sequence."""
    if not isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("in-process MCP server is only supported in library client")

    with make_mcp_server(tools=dependency_tools()) as mcp_server_info:
        tools = case["tools"]
        # Replace the placeholder URL with the actual server URL
        for tool in tools:
            if tool["type"] == "mcp" and tool["server_url"] == "<FILLED_BY_TEST_RUNNER>":
                tool["server_url"] = mcp_server_info["server_url"]

        stream = compat_client.responses.create(
            input=case["input"],
            model=text_model_id,
            tools=tools,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            chunks.append(chunk)

        # Should have at least response.created and response.completed
        assert len(chunks) >= 2, f"Expected at least 2 chunks (created + completed), got {len(chunks)}"

        # First chunk should be response.created
        assert chunks[0].type == "response.created", f"First chunk should be response.created, got {chunks[0].type}"

        # Last chunk should be response.completed
        assert chunks[-1].type == "response.completed", (
            f"Last chunk should be response.completed, got {chunks[-1].type}"
        )

        # Verify tool call streaming events are present
        chunk_types = [chunk.type for chunk in chunks]

        # Should have function call or MCP arguments delta/done events for tool calls
        delta_events = [
            chunk
            for chunk in chunks
            if chunk.type in ["response.function_call_arguments.delta", "response.mcp_call.arguments.delta"]
        ]
        done_events = [
            chunk
            for chunk in chunks
            if chunk.type in ["response.function_call_arguments.done", "response.mcp_call.arguments.done"]
        ]

        # Should have output item events for tool calls
        item_added_events = [chunk for chunk in chunks if chunk.type == "response.output_item.added"]
        item_done_events = [chunk for chunk in chunks if chunk.type == "response.output_item.done"]

        # Should have tool execution progress events
        mcp_in_progress_events = [chunk for chunk in chunks if chunk.type == "response.mcp_call.in_progress"]
        mcp_completed_events = [chunk for chunk in chunks if chunk.type == "response.mcp_call.completed"]

        # Verify we have substantial streaming activity (not just batch events)
        assert len(chunks) > 10, f"Expected rich streaming with many events, got only {len(chunks)} chunks"

        # Since this test involves MCP tool calls, we should see streaming events
        assert len(delta_events) > 0, (
            f"Expected function_call_arguments.delta or mcp_call.arguments.delta events, got chunk types: {chunk_types}"
        )
        assert len(done_events) > 0, (
            f"Expected function_call_arguments.done or mcp_call.arguments.done events, got chunk types: {chunk_types}"
        )

        # Should have output item events for function calls
        assert len(item_added_events) > 0, f"Expected response.output_item.added events, got chunk types: {chunk_types}"
        assert len(item_done_events) > 0, f"Expected response.output_item.done events, got chunk types: {chunk_types}"

        # Should have tool execution progress events
        assert len(mcp_in_progress_events) > 0, (
            f"Expected response.mcp_call.in_progress events, got chunk types: {chunk_types}"
        )
        assert len(mcp_completed_events) > 0, (
            f"Expected response.mcp_call.completed events, got chunk types: {chunk_types}"
        )
        # MCP failed events are optional (only if errors occur)

        # Verify progress events have proper structure
        for progress_event in mcp_in_progress_events:
            assert hasattr(progress_event, "item_id"), "Progress event should have 'item_id' field"
            assert hasattr(progress_event, "output_index"), "Progress event should have 'output_index' field"
            assert hasattr(progress_event, "sequence_number"), "Progress event should have 'sequence_number' field"

        for completed_event in mcp_completed_events:
            assert hasattr(completed_event, "sequence_number"), "Completed event should have 'sequence_number' field"

        # Verify delta events have proper structure
        for delta_event in delta_events:
            assert hasattr(delta_event, "delta"), "Delta event should have 'delta' field"
            assert hasattr(delta_event, "item_id"), "Delta event should have 'item_id' field"
            assert hasattr(delta_event, "sequence_number"), "Delta event should have 'sequence_number' field"
            assert delta_event.delta, "Delta should not be empty"

        # Verify done events have proper structure
        for done_event in done_events:
            assert hasattr(done_event, "arguments"), "Done event should have 'arguments' field"
            assert hasattr(done_event, "item_id"), "Done event should have 'item_id' field"
            assert done_event.arguments, "Final arguments should not be empty"

        # Verify output item added events have proper structure
        for added_event in item_added_events:
            assert hasattr(added_event, "item"), "Added event should have 'item' field"
            assert hasattr(added_event, "output_index"), "Added event should have 'output_index' field"
            assert hasattr(added_event, "sequence_number"), "Added event should have 'sequence_number' field"
            assert hasattr(added_event, "response_id"), "Added event should have 'response_id' field"
            assert added_event.item.type in ["function_call", "mcp_call"], "Added item should be a tool call"
            assert added_event.item.status == "in_progress", "Added item should be in progress"
            assert added_event.response_id, "Response ID should not be empty"
            assert isinstance(added_event.output_index, int), "Output index should be integer"
            assert added_event.output_index >= 0, "Output index should be non-negative"

        # Verify output item done events have proper structure
        for done_event in item_done_events:
            assert hasattr(done_event, "item"), "Done event should have 'item' field"
            assert hasattr(done_event, "output_index"), "Done event should have 'output_index' field"
            assert hasattr(done_event, "sequence_number"), "Done event should have 'sequence_number' field"
            assert hasattr(done_event, "response_id"), "Done event should have 'response_id' field"
            assert done_event.item.type in ["function_call", "mcp_call"], "Done item should be a tool call"
            # Note: MCP calls don't have a status field, only function calls do
            if done_event.item.type == "function_call":
                assert done_event.item.status == "completed", "Function call should be completed"
            assert done_event.response_id, "Response ID should not be empty"
            assert isinstance(done_event.output_index, int), "Output index should be integer"
            assert done_event.output_index >= 0, "Output index should be non-negative"

        # Group function call and MCP argument events by item_id (these should have proper tracking)
        argument_events_by_item_id = {}
        for chunk in chunks:
            if hasattr(chunk, "item_id") and chunk.type in [
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.mcp_call.arguments.delta",
                "response.mcp_call.arguments.done",
            ]:
                item_id = chunk.item_id
                if item_id not in argument_events_by_item_id:
                    argument_events_by_item_id[item_id] = []
                argument_events_by_item_id[item_id].append(chunk)

        for item_id, related_events in argument_events_by_item_id.items():
            # Should have at least one delta and one done event for a complete tool call
            delta_events = [
                e
                for e in related_events
                if e.type in ["response.function_call_arguments.delta", "response.mcp_call.arguments.delta"]
            ]
            done_events = [
                e
                for e in related_events
                if e.type in ["response.function_call_arguments.done", "response.mcp_call.arguments.done"]
            ]

            assert len(delta_events) > 0, f"Item {item_id} should have at least one delta event"
            assert len(done_events) == 1, f"Item {item_id} should have exactly one done event"

            # Verify all events have the same item_id
            for event in related_events:
                assert event.item_id == item_id, f"Event should have consistent item_id {item_id}, got {event.item_id}"

        # Verify content part events if they exist (for text streaming)
        content_part_added_events = [chunk for chunk in chunks if chunk.type == "response.content_part.added"]
        content_part_done_events = [chunk for chunk in chunks if chunk.type == "response.content_part.done"]

        # Content part events should be paired (if any exist)
        if len(content_part_added_events) > 0:
            assert len(content_part_done_events) > 0, (
                "Should have content_part.done events if content_part.added events exist"
            )

            # Verify content part event structure
            for added_event in content_part_added_events:
                assert hasattr(added_event, "response_id"), "Content part added event should have response_id"
                assert hasattr(added_event, "item_id"), "Content part added event should have item_id"
                assert hasattr(added_event, "part"), "Content part added event should have part"

                # TODO: enable this after the client types are updated
                # assert added_event.part.type == "output_text", "Content part should be an output_text"

            for done_event in content_part_done_events:
                assert hasattr(done_event, "response_id"), "Content part done event should have response_id"
                assert hasattr(done_event, "item_id"), "Content part done event should have item_id"
                assert hasattr(done_event, "part"), "Content part done event should have part"

                # TODO: enable this after the client types are updated
                # assert len(done_event.part.text) > 0, "Content part should have text when done"

        # Basic pairing check: each output_item.added should be followed by some activity
        # (but we can't enforce strict 1:1 pairing due to the complexity of multi-turn scenarios)
        assert len(item_added_events) > 0, "Should have at least one output_item.added event"

        # Verify response_id consistency across all events
        response_ids = set()
        for chunk in chunks:
            if hasattr(chunk, "response_id"):
                response_ids.add(chunk.response_id)
            elif hasattr(chunk, "response") and hasattr(chunk.response, "id"):
                response_ids.add(chunk.response.id)

        assert len(response_ids) == 1, f"All events should reference the same response_id, found: {response_ids}"

        # Get the final response from the last chunk
        final_chunk = chunks[-1]
        if hasattr(final_chunk, "response"):
            final_response = final_chunk.response

            # Verify multi-turn MCP tool execution results
            mcp_list_tools = [output for output in final_response.output if output.type == "mcp_list_tools"]
            mcp_calls = [output for output in final_response.output if output.type == "mcp_call"]
            message_outputs = [output for output in final_response.output if output.type == "message"]

            # Should have exactly 1 MCP list tools message (at the beginning)
            assert len(mcp_list_tools) == 1, f"Expected exactly 1 mcp_list_tools, got {len(mcp_list_tools)}"
            assert mcp_list_tools[0].server_label == "localmcp"
            assert len(mcp_list_tools[0].tools) == 5  # Updated for dependency tools
            expected_tool_names = {
                "get_user_id",
                "get_user_permissions",
                "check_file_access",
                "get_experiment_id",
                "get_experiment_results",
            }
            assert {t.name for t in mcp_list_tools[0].tools} == expected_tool_names

            # Should have at least 1 MCP call (the model should call at least one tool)
            assert len(mcp_calls) >= 1, f"Expected at least 1 mcp_call, got {len(mcp_calls)}"

            # All MCP calls should be completed (verifies our tool execution works)
            for mcp_call in mcp_calls:
                assert mcp_call.error is None, f"MCP call should not have errors, got: {mcp_call.error}"

            # Should have at least one final message response
            assert len(message_outputs) >= 1, f"Expected at least 1 message output, got {len(message_outputs)}"

            # Final message should be from assistant and completed
            final_message = message_outputs[-1]
            assert final_message.role == "assistant", (
                f"Final message should be from assistant, got {final_message.role}"
            )
            assert final_message.status == "completed", f"Final message should be completed, got {final_message.status}"
            assert len(final_message.content) > 0, "Final message should have content"

            # Check that the expected output appears in the response
            expected_output = case["output"]
            assert expected_output.lower() in final_response.output_text.lower(), (
                f"Expected '{expected_output}' to appear in response: {final_response.output_text}"
            )


@pytest.mark.parametrize(
    "text_format",
    # Not testing json_object because most providers don't actually support it.
    [
        {"type": "text"},
        {
            "type": "json_schema",
            "name": "capitals",
            "description": "A schema for the capital of each country",
            "schema": {"type": "object", "properties": {"capital": {"type": "string"}}},
            "strict": True,
        },
    ],
)
def test_response_text_format(compat_client, text_model_id, text_format):
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("Responses API text format is not yet supported in library client.")

    stream = False
    response = compat_client.responses.create(
        model=text_model_id,
        input="What is the capital of France?",
        stream=stream,
        text={"format": text_format},
    )
    # by_alias=True is needed because otherwise Pydantic renames our "schema" field
    assert response.text.format.model_dump(exclude_none=True, by_alias=True) == text_format
    assert "paris" in response.output_text.lower()
    if text_format["type"] == "json_schema":
        assert "paris" in json.loads(response.output_text)["capital"].lower()


@pytest.fixture
def vector_store_with_filtered_files(compat_client, text_model_id, tmp_path_factory):
    """Create a vector store with multiple files that have different attributes for filtering tests."""
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("Responses API file search is not yet supported in library client.")

    vector_store = _new_vector_store(compat_client, "test_vector_store_with_filters")
    tmp_path = tmp_path_factory.mktemp("filter_test_files")

    # Create multiple files with different attributes
    files_data = [
        {
            "name": "us_marketing_q1.txt",
            "content": "US promotional campaigns for Q1 2023. Revenue increased by 15% in the US region.",
            "attributes": {
                "region": "us",
                "category": "marketing",
                "date": 1672531200,  # Jan 1, 2023
            },
        },
        {
            "name": "us_engineering_q2.txt",
            "content": "US technical updates for Q2 2023. New features deployed in the US region.",
            "attributes": {
                "region": "us",
                "category": "engineering",
                "date": 1680307200,  # Apr 1, 2023
            },
        },
        {
            "name": "eu_marketing_q1.txt",
            "content": "European advertising campaign results for Q1 2023. Strong growth in EU markets.",
            "attributes": {
                "region": "eu",
                "category": "marketing",
                "date": 1672531200,  # Jan 1, 2023
            },
        },
        {
            "name": "asia_sales_q3.txt",
            "content": "Asia Pacific revenue figures for Q3 2023. Record breaking quarter in Asia.",
            "attributes": {
                "region": "asia",
                "category": "sales",
                "date": 1688169600,  # Jul 1, 2023
            },
        },
    ]

    file_ids = []
    for file_data in files_data:
        # Create file
        file_path = tmp_path / file_data["name"]
        file_path.write_text(file_data["content"])

        # Upload file
        file_response = _upload_file(compat_client, file_data["name"], str(file_path))
        file_ids.append(file_response.id)

        # Attach file to vector store with attributes
        file_attach_response = compat_client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=file_response.id,
            attributes=file_data["attributes"],
        )

        # Wait for attachment
        while file_attach_response.status == "in_progress":
            time.sleep(0.1)
            file_attach_response = compat_client.vector_stores.files.retrieve(
                vector_store_id=vector_store.id,
                file_id=file_response.id,
            )
        assert file_attach_response.status == "completed"

    yield vector_store

    # Cleanup: delete vector store and files
    try:
        compat_client.vector_stores.delete(vector_store_id=vector_store.id)
        for file_id in file_ids:
            try:
                compat_client.files.delete(file_id=file_id)
            except Exception:
                pass  # File might already be deleted
    except Exception:
        pass  # Best effort cleanup


def test_response_file_search_filter_by_region(compat_client, text_model_id, vector_store_with_filtered_files):
    """Test file search with region equality filter."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {"type": "eq", "key": "region", "value": "us"},
        }
    ]

    response = compat_client.responses.create(
        model=text_model_id,
        input="What are the updates from the US region?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    # Verify file search was called with US filter
    assert len(response.output) > 1
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return US files (not EU or Asia files)
    for result in response.output[0].results:
        assert "us" in result.text.lower() or "US" in result.text
        # Ensure non-US regions are NOT returned
        assert "european" not in result.text.lower()
        assert "asia" not in result.text.lower()


def test_response_file_search_filter_by_category(compat_client, text_model_id, vector_store_with_filtered_files):
    """Test file search with category equality filter."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {"type": "eq", "key": "category", "value": "marketing"},
        }
    ]

    response = compat_client.responses.create(
        model=text_model_id,
        input="Show me all marketing reports",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return marketing files (not engineering or sales)
    for result in response.output[0].results:
        # Marketing files should have promotional/advertising content
        assert "promotional" in result.text.lower() or "advertising" in result.text.lower()
        # Ensure non-marketing categories are NOT returned
        assert "technical" not in result.text.lower()
        assert "revenue figures" not in result.text.lower()


def test_response_file_search_filter_by_date_range(compat_client, text_model_id, vector_store_with_filtered_files):
    """Test file search with date range filter using compound AND."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {
                "type": "and",
                "filters": [
                    {
                        "type": "gte",
                        "key": "date",
                        "value": 1672531200,  # Jan 1, 2023
                    },
                    {
                        "type": "lt",
                        "key": "date",
                        "value": 1680307200,  # Apr 1, 2023
                    },
                ],
            },
        }
    ]

    response = compat_client.responses.create(
        model=text_model_id,
        input="What happened in Q1 2023?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return Q1 files (not Q2 or Q3)
    for result in response.output[0].results:
        assert "q1" in result.text.lower()
        # Ensure non-Q1 quarters are NOT returned
        assert "q2" not in result.text.lower()
        assert "q3" not in result.text.lower()


def test_response_file_search_filter_compound_and(compat_client, text_model_id, vector_store_with_filtered_files):
    """Test file search with compound AND filter (region AND category)."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "region", "value": "us"},
                    {"type": "eq", "key": "category", "value": "engineering"},
                ],
            },
        }
    ]

    response = compat_client.responses.create(
        model=text_model_id,
        input="What are the engineering updates from the US?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return US engineering files
    assert len(response.output[0].results) >= 1
    for result in response.output[0].results:
        assert "us" in result.text.lower() and "technical" in result.text.lower()
        # Ensure it's not from other regions or categories
        assert "european" not in result.text.lower() and "asia" not in result.text.lower()
        assert "promotional" not in result.text.lower() and "revenue" not in result.text.lower()


def test_response_file_search_filter_compound_or(compat_client, text_model_id, vector_store_with_filtered_files):
    """Test file search with compound OR filter (marketing OR sales)."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {
                "type": "or",
                "filters": [
                    {"type": "eq", "key": "category", "value": "marketing"},
                    {"type": "eq", "key": "category", "value": "sales"},
                ],
            },
        }
    ]

    response = compat_client.responses.create(
        model=text_model_id,
        input="Show me marketing and sales documents",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should return marketing and sales files, but NOT engineering
    categories_found = set()
    for result in response.output[0].results:
        text_lower = result.text.lower()
        if "promotional" in text_lower or "advertising" in text_lower:
            categories_found.add("marketing")
        if "revenue figures" in text_lower:
            categories_found.add("sales")
        # Ensure engineering files are NOT returned
        assert "technical" not in text_lower, f"Engineering file should not be returned, but got: {result.text}"

    # Verify we got at least one of the expected categories
    assert len(categories_found) > 0, "Should have found at least one marketing or sales file"
    assert categories_found.issubset({"marketing", "sales"}), f"Found unexpected categories: {categories_found}"
