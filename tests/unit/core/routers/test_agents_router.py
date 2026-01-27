# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException

from llama_stack.core.server.fastapi_router_registry import build_fastapi_router
from llama_stack_api import Agents, Api
from llama_stack_api.agents.models import (
    CreateResponseRequest,
    DeleteResponseRequest,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    RetrieveResponseRequest,
)
from llama_stack_api.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
)


def test_openapi_create_response_advertises_json_and_sse_200():
    """Regression test for OpenAPI shape of POST /v1/responses.

    We expect:
    - 200 response (not 204)
    - both JSON and SSE content types documented
    """

    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    schema = app.openapi()

    post = schema["paths"]["/v1/responses"]["post"]
    responses = post["responses"]
    assert "200" in responses
    assert "204" not in responses

    content = responses["200"]["content"]
    assert "application/json" in content
    assert "text/event-stream" in content

    assert content["application/json"]["schema"]["$ref"] == "#/components/schemas/OpenAIResponseObject"
    assert content["text/event-stream"]["schema"]["$ref"] == "#/components/schemas/OpenAIResponseObjectStream"


async def test_create_response_returns_sse_streaming_response_when_impl_streams():
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    async def _stream():
        yield {"type": "response.output_text.delta", "delta": "hello"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=True)
    response = await create(request)

    assert response.media_type == "text/event-stream"


async def test_create_response_maps_value_error_to_400_http_exception():
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.create_openai_response.side_effect = ValueError("not found")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=False)

    with pytest.raises(HTTPException) as excinfo:
        await create(request)

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "not found"


async def test_create_response_returns_json_for_non_streaming():
    """Test POST /v1/responses returns OpenAIResponseObject when stream=False."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    expected_response = OpenAIResponseObject(
        id="resp_123",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=False)
    response = await create(request)

    assert not hasattr(response, "media_type")  # Not a StreamingResponse
    assert response.id == "resp_123"
    assert response.status == "completed"


async def test_sse_format_is_correct():
    """Test that streaming responses produce valid SSE format."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    async def _stream():
        yield {"type": "response.output_text.delta", "delta": "hello"}
        yield {"type": "response.completed"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=True)
    response = await create(request)

    # Collect SSE events from the body iterator
    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)

    assert len(events) == 2
    # Verify SSE format: "data: {...}\n\n"
    assert events[0].startswith("data: ")
    assert events[0].endswith("\n\n")
    assert '"type": "response.output_text.delta"' in events[0]


async def test_sse_stream_keeps_provider_context():
    from llama_stack.core.request_headers import PROVIDER_DATA_VAR

    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    provider_data = {"provider": "test"}

    async def _stream():
        yield {"provider_data": PROVIDER_DATA_VAR.get()}
        yield {"type": "response.completed"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    token = PROVIDER_DATA_VAR.set(provider_data)
    try:
        request = CreateResponseRequest(input="hi", model="test", stream=True)
        response = await create(request)
    finally:
        PROVIDER_DATA_VAR.reset(token)

    first_event = None
    async for chunk in response.body_iterator:
        first_event = chunk
        break

    assert first_event is not None
    assert '"provider_data": {"provider": "test"}' in first_event


async def test_sse_stream_reports_value_error_as_http_exception():
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    async def _stream():
        raise ValueError("not found")
        yield {"type": "response.completed"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=True)
    response = await create(request)

    first_event = None
    async for chunk in response.body_iterator:
        first_event = chunk
        break

    assert first_event is not None
    assert '"status_code": 400' in first_event
    assert '"message": "not found"' in first_event


async def test_get_response_returns_response_object():
    """Test GET /v1/responses/{response_id} returns OpenAIResponseObject."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    expected_response = OpenAIResponseObject(
        id="resp_123",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.get_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    get_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}" and "GET" in getattr(r, "methods", set())
    )

    request = RetrieveResponseRequest(response_id="resp_123")
    response = await get_endpoint(request)

    assert response.id == "resp_123"
    assert response.status == "completed"
    impl.get_openai_response.assert_awaited_once()


async def test_get_response_maps_value_error_to_400():
    """Test GET /v1/responses/{response_id} maps ValueError to HTTP 400."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.get_openai_response.side_effect = ValueError("Response not found")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    get_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}" and "GET" in getattr(r, "methods", set())
    )

    request = RetrieveResponseRequest(response_id="nonexistent")

    with pytest.raises(HTTPException) as excinfo:
        await get_endpoint(request)

    assert excinfo.value.status_code == 400
    assert "not found" in excinfo.value.detail.lower()


async def test_list_responses_returns_list():
    """Test GET /v1/responses returns ListOpenAIResponseObject."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    expected_response = ListOpenAIResponseObject(
        object="list",
        data=[],
        has_more=False,
        first_id="resp_first",
        last_id="resp_last",
    )
    impl.list_openai_responses.return_value = expected_response

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    list_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "GET" in getattr(r, "methods", set())
    )

    request = ListResponsesRequest()
    response = await list_endpoint(request)

    assert response.object == "list"
    assert response.has_more is False
    impl.list_openai_responses.assert_awaited_once()


async def test_list_input_items_returns_items():
    """Test GET /v1/responses/{response_id}/input_items returns input items."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    expected_response = ListOpenAIResponseInputItem(
        object="list",
        data=[],
    )
    impl.list_openai_response_input_items.return_value = expected_response

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    list_endpoint = next(
        r.endpoint for r in router.routes if getattr(r, "path", None) == "/v1/responses/{response_id}/input_items"
    )

    request = ListResponseInputItemsRequest(response_id="resp_123")
    response = await list_endpoint(request)

    assert response.object == "list"
    impl.list_openai_response_input_items.assert_awaited_once()


async def test_delete_response_returns_confirmation():
    """Test DELETE /v1/responses/{response_id} returns deletion confirmation."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    expected_response = OpenAIDeleteResponseObject(
        id="resp_123",
        object="response",
        deleted=True,
    )
    impl.delete_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    delete_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}" and "DELETE" in getattr(r, "methods", set())
    )

    request = DeleteResponseRequest(response_id="resp_123")
    response = await delete_endpoint(request)

    assert response.id == "resp_123"
    assert response.deleted is True
    impl.delete_openai_response.assert_awaited_once()


async def test_delete_response_maps_value_error_to_400():
    """Test DELETE /v1/responses/{response_id} maps ValueError to HTTP 400."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.delete_openai_response.side_effect = ValueError("Response not found")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    delete_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}" and "DELETE" in getattr(r, "methods", set())
    )

    request = DeleteResponseRequest(response_id="nonexistent")

    with pytest.raises(HTTPException) as excinfo:
        await delete_endpoint(request)

    assert excinfo.value.status_code == 400
