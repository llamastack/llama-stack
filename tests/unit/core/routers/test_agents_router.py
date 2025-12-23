# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from llama_stack.core.server.fastapi_router_registry import build_fastapi_router
from llama_stack_api import Agents, Api


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

    from llama_stack_api.agents.models import CreateResponseRequest

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

    from llama_stack_api.agents.models import CreateResponseRequest

    request = CreateResponseRequest(input="hi", model="test", stream=False)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        await create(request)

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "not found"
