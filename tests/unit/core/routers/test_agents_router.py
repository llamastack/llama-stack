# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from fastapi import FastAPI

from llama_stack_api import Agents
from llama_stack_api.agents.fastapi_routes import create_router


def test_openapi_create_response_advertises_json_and_sse_200():
    """
    Regression test for the OpenAPI shape of POST /v1/responses.

    We expect:
    - 200 response (not 204)
    - both JSON and SSE content types documented
    """

    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    app.include_router(create_router(impl))

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
