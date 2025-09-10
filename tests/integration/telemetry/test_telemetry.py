# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from llama_stack_client import Agent


@pytest.fixture(scope="module", autouse=True)
def setup_telemetry_data(llama_stack_client, text_model_id):
    """Setup fixture that creates telemetry data before tests run."""
    agent = Agent(llama_stack_client, model=text_model_id, instructions="You are a helpful assistant")

    session_id = agent.create_session(f"test-setup-session-{uuid4()}")

    messages = [
        "What is 2 + 2?",
        "Tell me a short joke",
    ]

    for msg in messages:
        agent.create_turn(
            messages=[{"role": "user", "content": msg}],
            session_id=session_id,
            stream=False,
        )

    for i in range(2):
        llama_stack_client.inference.chat_completion(
            model_id=text_model_id, messages=[{"role": "user", "content": f"Test trace {i}"}]
        )

    start_time = time.time()

    while time.time() - start_time < 30:
        traces = llama_stack_client.telemetry.query_traces(limit=10)
        if len(traces) >= 4:
            break
        time.sleep(1)

    if len(traces) < 4:
        pytest.fail(f"Failed to create sufficient telemetry data after 30s. Got {len(traces)} traces.")

    # Wait for 5 seconds to ensure traces has completed logging
    time.sleep(5)

    yield


def test_query_traces_basic(llama_stack_client):
    """Test basic trace querying functionality with proper data validation."""
    all_traces = llama_stack_client.telemetry.query_traces(limit=5)

    assert isinstance(all_traces, list), "Should return a list of traces"
    assert len(all_traces) >= 4, "Should have at least 4 traces from setup"

    # Verify trace structure and data quality
    first_trace = all_traces[0]
    assert hasattr(first_trace, "trace_id"), "Trace should have trace_id"
    assert hasattr(first_trace, "start_time"), "Trace should have start_time"
    assert hasattr(first_trace, "root_span_id"), "Trace should have root_span_id"

    # Validate trace_id is a valid UUID format
    assert isinstance(first_trace.trace_id, str) and len(first_trace.trace_id) > 0, (
        "trace_id should be non-empty string"
    )

    # Validate start_time format and not in the future
    now = datetime.now(UTC)
    if isinstance(first_trace.start_time, str):
        trace_time = datetime.fromisoformat(first_trace.start_time.replace("Z", "+00:00"))
    else:
        # start_time is already a datetime object
        trace_time = first_trace.start_time
        if trace_time.tzinfo is None:
            trace_time = trace_time.replace(tzinfo=UTC)

    # Ensure trace time is not in the future (but allow any age in the past for persistent test data)
    time_diff = (now - trace_time).total_seconds()
    assert time_diff >= 0, f"Trace start_time should not be in the future, got {time_diff}s"

    # Validate root_span_id exists and is non-empty
    assert isinstance(first_trace.root_span_id, str) and len(first_trace.root_span_id) > 0, (
        "root_span_id should be non-empty string"
    )

    # Test querying specific trace by ID
    specific_trace = llama_stack_client.telemetry.get_trace(trace_id=first_trace.trace_id)
    assert specific_trace.trace_id == first_trace.trace_id, "Retrieved trace should match requested ID"
    assert specific_trace.start_time == first_trace.start_time, "Retrieved trace should have same start_time"
    assert specific_trace.root_span_id == first_trace.root_span_id, "Retrieved trace should have same root_span_id"

    # Test pagination with proper validation
    recent_traces = llama_stack_client.telemetry.query_traces(limit=3, offset=0)
    assert len(recent_traces) <= 3, "Should return at most 3 traces when limit=3"
    assert len(recent_traces) >= 1, "Should return at least 1 trace"

    # Verify all traces have required fields
    for trace in recent_traces:
        assert hasattr(trace, "trace_id") and trace.trace_id, "Each trace should have non-empty trace_id"
        assert hasattr(trace, "start_time") and trace.start_time, "Each trace should have non-empty start_time"
        assert hasattr(trace, "root_span_id") and trace.root_span_id, "Each trace should have non-empty root_span_id"


def test_query_spans_basic(llama_stack_client):
    """Test basic span querying functionality with proper validation."""
    spans = llama_stack_client.telemetry.query_spans(attribute_filters=[], attributes_to_return=[])

    assert isinstance(spans, list), "Should return a list of spans"
    assert len(spans) >= 1, "Should have at least one span from setup"

    # Verify span structure and data quality
    first_span = spans[0]
    required_attrs = ["span_id", "name", "trace_id"]
    for attr in required_attrs:
        assert hasattr(first_span, attr), f"Span should have {attr} attribute"
        assert getattr(first_span, attr), f"Span {attr} should not be empty"

    # Validate span data types and content
    assert isinstance(first_span.span_id, str) and len(first_span.span_id) > 0, "span_id should be non-empty string"
    assert isinstance(first_span.name, str) and len(first_span.name) > 0, "span name should be non-empty string"
    assert isinstance(first_span.trace_id, str) and len(first_span.trace_id) > 0, "trace_id should be non-empty string"

    # Verify span belongs to a valid trace (test with traces we know exist)
    all_traces = llama_stack_client.telemetry.query_traces(limit=10)
    trace_ids = {t.trace_id for t in all_traces}
    if first_span.trace_id in trace_ids:
        trace = llama_stack_client.telemetry.get_trace(trace_id=first_span.trace_id)
        assert trace is not None, "Should be able to retrieve trace for valid trace_id"
        assert trace.trace_id == first_span.trace_id, "Trace ID should match span's trace_id"

    # Test with span filtering and validate results
    filtered_spans = llama_stack_client.telemetry.query_spans(
        attribute_filters=[{"key": "name", "op": "eq", "value": first_span.name}],
        attributes_to_return=["name", "span_id"],
    )
    assert isinstance(filtered_spans, list), "Should return a list with span name filter"

    # Validate filtered spans if filtering works
    if len(filtered_spans) > 0:
        for span in filtered_spans:
            assert hasattr(span, "name"), "Filtered spans should have name attribute"
            assert hasattr(span, "span_id"), "Filtered spans should have span_id attribute"
            assert span.name == first_span.name, "Filtered spans should match the filter criteria"
            assert isinstance(span.span_id, str) and len(span.span_id) > 0, "Filtered span_id should be valid"

    # Test that all spans have consistent structure
    for span in spans:
        for attr in required_attrs:
            assert hasattr(span, attr) and getattr(span, attr), f"All spans should have non-empty {attr}"


def test_telemetry_pagination(llama_stack_client):
    """Test pagination in telemetry queries."""
    # Get total count of traces
    all_traces = llama_stack_client.telemetry.query_traces(limit=20)
    total_count = len(all_traces)
    assert total_count >= 4, "Should have at least 4 traces from setup"

    # Test trace pagination
    page1 = llama_stack_client.telemetry.query_traces(limit=2, offset=0)
    page2 = llama_stack_client.telemetry.query_traces(limit=2, offset=2)

    assert len(page1) == 2, "First page should have exactly 2 traces"
    assert len(page2) >= 1, "Second page should have at least 1 trace"

    # Verify no overlap between pages
    page1_ids = {t.trace_id for t in page1}
    page2_ids = {t.trace_id for t in page2}
    assert len(page1_ids.intersection(page2_ids)) == 0, "Pages should contain different traces"

    # Test ordering
    ordered_traces = llama_stack_client.telemetry.query_traces(limit=5, order_by=["start_time"])
    assert len(ordered_traces) >= 4, "Should have at least 4 traces for ordering test"

    # Verify ordering by start_time
    for i in range(len(ordered_traces) - 1):
        current_time = ordered_traces[i].start_time
        next_time = ordered_traces[i + 1].start_time
        assert current_time <= next_time, f"Traces should be ordered by start_time: {current_time} > {next_time}"

    # Test limit behavior
    limited = llama_stack_client.telemetry.query_traces(limit=3)
    assert len(limited) == 3, "Should return exactly 3 traces when limit=3"
