# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import pytest

from .fixtures.test_cases import basic_test_cases, image_test_cases, multi_turn_image_test_cases, multi_turn_test_cases
from .streaming_assertions import StreamingValidator


@pytest.mark.parametrize("case", basic_test_cases)
def test_response_non_streaming_basic(compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case.input,
        stream=False,
    )
    output_text = response.output_text.lower().strip()
    assert len(output_text) > 0
    assert case.expected.lower() in output_text

    # Verify usage is reported
    assert response.usage is not None, "Response should include usage information"
    assert response.usage.input_tokens > 0, "Input tokens should be greater than 0"
    assert response.usage.output_tokens > 0, "Output tokens should be greater than 0"
    assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens, (
        "Total tokens should equal input + output tokens"
    )

    retrieved_response = compat_client.responses.retrieve(response_id=response.id)
    assert retrieved_response.output_text == response.output_text

    next_response = compat_client.responses.create(
        model=text_model_id,
        input="Repeat your previous response in all caps.",
        previous_response_id=response.id,
    )
    next_output_text = next_response.output_text.strip()
    assert case.expected.upper() in next_output_text


@pytest.mark.parametrize("case", basic_test_cases)
def test_response_streaming_basic(compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case.input,
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
            assert event_times[0] < 0.2, (
                f"response.created should be emitted immediately (took {event_times[0]} seconds)"
            )
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
            assert case.expected.lower() in output_text, f"Expected '{case.expected}' in response"

            # Verify usage is reported in final response
            assert chunk.response.usage is not None, "Completed response should include usage information"
            assert chunk.response.usage.input_tokens > 0, "Input tokens should be greater than 0"
            assert chunk.response.usage.output_tokens > 0, "Output tokens should be greater than 0"
            assert (
                chunk.response.usage.total_tokens
                == chunk.response.usage.input_tokens + chunk.response.usage.output_tokens
            ), "Total tokens should equal input + output tokens"

    # Use validator for common checks
    validator = StreamingValidator(events)
    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()

    # Verify stored response matches streamed response
    retrieved_response = compat_client.responses.retrieve(response_id=response_id)
    final_event = events[-1]
    assert retrieved_response.output_text == final_event.response.output_text


@pytest.mark.parametrize("case", basic_test_cases)
def test_response_streaming_incremental_content(compat_client, text_model_id, case):
    """Test that streaming actually delivers content incrementally, not just at the end."""
    response = compat_client.responses.create(
        model=text_model_id,
        input=case.input,
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

    validator = StreamingValidator(events)
    validator.assert_basic_event_sequence()

    # Check if we have incremental content updates
    event_types = [event.type for event in events]
    created_index = event_types.index("response.created")
    completed_index = event_types.index("response.completed")

    # The key test: verify content progression
    created_content = content_snapshots[created_index]
    completed_content = content_snapshots[completed_index]

    # Verify that response.created has empty or minimal content
    assert len(created_content) == 0, f"response.created should have empty content, got: {repr(created_content[:100])}"

    # Verify that response.completed has the full content
    assert len(completed_content) > 0, "response.completed should have content"
    assert case.expected.lower() in completed_content.lower(), f"Expected '{case.expected}' in final content"

    # Use validator for incremental content checks
    delta_content_total = validator.assert_has_incremental_content()

    # Verify that the accumulated delta content matches the final content
    assert delta_content_total.strip() == completed_content.strip(), (
        f"Delta content '{delta_content_total}' should match final content '{completed_content}'"
    )

    # Verify timing: delta events should come between created and completed
    delta_events = [i for i, event_type in enumerate(event_types) if event_type == "response.output_text.delta"]
    for delta_idx in delta_events:
        assert created_index < delta_idx < completed_index, (
            f"Delta event at index {delta_idx} should be between created ({created_index}) and completed ({completed_index})"
        )


@pytest.mark.parametrize("case", multi_turn_test_cases)
def test_response_non_streaming_multi_turn(compat_client, text_model_id, case):
    previous_response_id = None
    for turn_input, turn_expected in case.turns:
        response = compat_client.responses.create(
            model=text_model_id,
            input=turn_input,
            previous_response_id=previous_response_id,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn_expected.lower() in output_text


@pytest.mark.parametrize("case", image_test_cases)
def test_response_non_streaming_image(compat_client, text_model_id, case):
    response = compat_client.responses.create(
        model=text_model_id,
        input=case.input,
        stream=False,
    )
    output_text = response.output_text.lower()
    assert case.expected.lower() in output_text


@pytest.mark.parametrize("case", multi_turn_image_test_cases)
def test_response_non_streaming_multi_turn_image(compat_client, text_model_id, case):
    previous_response_id = None
    for turn_input, turn_expected in case.turns:
        response = compat_client.responses.create(
            model=text_model_id,
            input=turn_input,
            previous_response_id=previous_response_id,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn_expected.lower() in output_text
