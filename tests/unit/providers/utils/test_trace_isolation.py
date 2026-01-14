# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test demonstrating TraceContext span isolation bug and fix.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import Mock

from llama_stack.apis.telemetry import Span
from llama_stack.providers.utils.telemetry.tracing import TraceContext, generate_span_id


class BuggyTraceContext:
    """
    Before fix: This recreates the buggy behavior of TraceContext where spans was a class variable.
    """

    spans: list[Span] = []  # Class variable - shared across all instances

    def __init__(self, logger, trace_id: str):
        self.logger = logger
        self.trace_id = trace_id

    def push_span(self, name: str):
        """Simplified push_span to demonstrate the bug"""
        span = Span(
            span_id=generate_span_id(),
            trace_id=self.trace_id,
            name=name,
            start_time=datetime.now(UTC),
        )
        self.spans.append(span)
        return span


async def simulate_user_request(user_id: int, context):
    """Simulate a user making 2 operations"""
    context.push_span(f"user{user_id}-operation-1")
    context.push_span(f"user{user_id}-operation-2")


async def test_buggy_version():
    """Test before fix - demonstrates the bug with class variable"""
    BuggyTraceContext.spans = []
    mock_logger = Mock()

    trace1 = BuggyTraceContext(mock_logger, "trace-user-1")
    trace2 = BuggyTraceContext(mock_logger, "trace-user-2")

    await asyncio.gather(
        simulate_user_request(1, trace1),
        simulate_user_request(2, trace2),
    )

    return trace1, trace2


async def test_fixed_version():
    """Test after fix - uses fixed TraceContext"""
    mock_logger = Mock()

    trace1 = TraceContext(mock_logger, "trace-user-1")
    trace2 = TraceContext(mock_logger, "trace-user-2")

    await asyncio.gather(
        simulate_user_request(1, trace1),
        simulate_user_request(2, trace2),
    )

    return trace1, trace2


async def test_trace_context_span_isolation():
    """
    Test that demonstrates the TraceContext span isolation bug and validates the fix.

    Before fix: When spans was a class variable, all TraceContext instances
    shared the same list object, causing all spans from all users to be mixed
    together in one shared list.

    After fix: When spans is an instance variable, each TraceContext instance
    has its own isolated list of spans.
    """
    # Test before fix - demonstrates the bug
    trace1_buggy, trace2_buggy = await test_buggy_version()

    # Verify buggy behavior - both instances share the same list object
    assert trace1_buggy.spans is trace2_buggy.spans, "Buggy version should share the same spans list object"
    assert len(trace1_buggy.spans) == 4, "Buggy version: both traces see all 4 spans"

    # Both traces see spans from both trace ids mixed together
    buggy_trace_ids = {s.trace_id for s in trace1_buggy.spans}
    assert buggy_trace_ids == {"trace-user-1", "trace-user-2"}, "Buggy version mixes spans from different trace IDs"

    # Test after fix - validates the fix with actual TraceContext
    trace1_fixed, trace2_fixed = await test_fixed_version()

    # Verify fixed behavior - each instance has its own list object
    assert trace1_fixed.spans is not trace2_fixed.spans, "Fixed version should have separate spans lists"
    assert len(trace1_fixed.spans) == 2, "Fixed version: trace-user-1 has only its 2 spans"
    assert len(trace2_fixed.spans) == 2, "Fixed version: trace-user-2 has only its 2 spans"

    # Each trace sees only its own trace ID
    fixed_trace_ids_1 = {s.trace_id for s in trace1_fixed.spans}
    fixed_trace_ids_2 = {s.trace_id for s in trace2_fixed.spans}
    assert fixed_trace_ids_1 == {"trace-user-1"}, "Fixed version: trace-user-1 sees only its own spans"
    assert fixed_trace_ids_2 == {"trace-user-2"}, "Fixed version: trace-user-2 sees only its own spans"
