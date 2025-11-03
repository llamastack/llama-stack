# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging  # allow-direct-logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.core.server.server import create_dynamic_typed_route, create_sse_event, sse_generator


@pytest.fixture
def suppress_sse_errors(caplog):
    """Suppress expected ERROR logs for tests that deliberately trigger SSE errors"""
    caplog.set_level(logging.CRITICAL, logger="llama_stack.core.server.server")


async def test_sse_generator_basic():
    # An AsyncIterator wrapped in an Awaitable, just like our web methods
    async def async_event_gen():
        async def event_gen():
            yield "Test event 1"
            yield "Test event 2"

        return event_gen()

    sse_gen = sse_generator(async_event_gen())
    assert sse_gen is not None

    # Test that the events are streamed correctly
    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)
    assert len(seen_events) == 2
    assert seen_events[0] == create_sse_event("Test event 1")
    assert seen_events[1] == create_sse_event("Test event 2")


async def test_sse_generator_client_disconnected():
    # An AsyncIterator wrapped in an Awaitable, just like our web methods
    async def async_event_gen():
        async def event_gen():
            yield "Test event 1"
            # Simulate a client disconnect before emitting event 2
            raise asyncio.CancelledError()

        return event_gen()

    sse_gen = sse_generator(async_event_gen())
    assert sse_gen is not None

    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)

    # We should see 1 event before the client disconnected
    assert len(seen_events) == 1
    assert seen_events[0] == create_sse_event("Test event 1")


async def test_sse_generator_client_disconnected_before_response_starts():
    # Disconnect before the response starts
    async def async_event_gen():
        raise asyncio.CancelledError()

    sse_gen = sse_generator(async_event_gen())
    assert sse_gen is not None

    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)

    # No events should be seen since the client disconnected immediately
    assert len(seen_events) == 0


async def test_sse_generator_error_before_response_starts(suppress_sse_errors):
    # Raise an error before the response starts
    async def async_event_gen():
        raise Exception("Test error")

    sse_gen = sse_generator(async_event_gen())
    assert sse_gen is not None

    seen_events = []
    async for event in sse_gen:
        seen_events.append(event)

    # We should have 1 error event
    assert len(seen_events) == 1
    assert 'data: {"error":' in seen_events[0]


async def test_paginated_response_url_setting():
    """Test that PaginatedResponse gets url set to route path."""

    async def mock_api_method():
        return PaginatedResponse(data=[], has_more=False, url=None)

    route_handler = create_dynamic_typed_route(mock_api_method, "get", "/test/route")

    # Mock minimal request with proper state object
    request = MagicMock()
    request.scope = {"user_attributes": {}, "principal": ""}
    request.headers = {}
    request.body = AsyncMock(return_value=b"")

    # Create a simple state object without auto-generating attributes
    class MockState:
        pass

    request.state = MockState()

    result = await route_handler(request)

    assert isinstance(result, PaginatedResponse)
    assert result.url == "/test/route"
