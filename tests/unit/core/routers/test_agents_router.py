# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from fastapi import FastAPI
from starlette.testclient import TestClient

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


def test_create_response_maps_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError to HTTP 400."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.create_openai_response.side_effect = ValueError("not found")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "not found"


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


def test_get_response_maps_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError on GET to HTTP 400."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.get_openai_response.side_effect = ValueError("Response not found")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/v1/responses/nonexistent")

    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


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


def test_delete_response_maps_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError on DELETE to HTTP 400."""
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.delete_openai_response.side_effect = ValueError("Response not found")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.delete("/v1/responses/nonexistent")

    assert resp.status_code == 400


def test_request_validation_error_passes_through_route_class():
    """RequestValidationError must NOT be caught by _ExceptionTranslatingRoute.

    FastAPI has its own handler for RequestValidationError that returns a
    422 response with detailed validation errors.  The route class must let
    it pass through so that invalid request bodies (e.g. max_tool_calls=0)
    get a proper 422 instead of a generic 500.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Agents)

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)

    # max_tool_calls has ge=1 constraint — sending 0 triggers RequestValidationError
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False, "max_tool_calls": 0})

    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    # FastAPI returns a list of validation errors
    assert isinstance(body["detail"], list)
    assert any("max_tool_calls" in str(err) for err in body["detail"])


def test_exception_translating_route_converts_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError to HTTP 400.

    ValueError (including pydantic.ValidationError) is translated to a
    400 response by the route class.  This test exercises the route class
    via TestClient (full ASGI stack) to verify end-to-end behavior.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.create_openai_response.side_effect = ValueError("bad input value")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})

    assert resp.status_code == 400
    assert resp.headers["content-type"] == "application/json"
    assert resp.json()["detail"] == "bad input value"


def test_unknown_exception_propagates_to_global_handler():
    """Unknown exception types (e.g. RuntimeError) propagate past the route class.

    The route class only translates known types (ValueError, LlamaStackError).
    Unknown exceptions are left for the server's global exception handler,
    which uses the full translate_exception pipeline from llama_stack.core.
    """
    from llama_stack.core.server.server import global_exception_handler

    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    impl = AsyncMock(spec=Agents)
    impl.create_openai_response.side_effect = RuntimeError("something broke")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})

    assert resp.status_code == 500
    # Global handler returns {"error": {"detail": ...}}
    assert "error" in resp.json()


def test_consecutive_value_errors_keep_connection_alive():
    """Consecutive ValueError responses must not break the HTTP keep-alive connection.

    The route class converts ValueError to HTTPException at the route level.
    FastAPI's built-in handler processes HTTPException cleanly — no re-raise,
    no transport close, no broken connections.

    This test sends two error-producing requests on the **same** TestClient
    (which reuses the underlying connection) and verifies both get proper
    JSON error responses.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.create_openai_response.side_effect = ValueError("bad request")

    router = build_fastapi_router(Api.agents, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)

    # First request — triggers the error path
    resp1 = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})
    assert resp1.status_code == 400

    # Second request on the same connection — must NOT get a connection error
    resp2 = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "bad request"
