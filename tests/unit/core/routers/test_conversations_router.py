# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from fastapi import FastAPI
from starlette.testclient import TestClient

from llama_stack.core.server.fastapi_router_registry import build_fastapi_router
from llama_stack.core.server.server import global_exception_handler
from llama_stack_api import Api
from llama_stack_api.common.errors import ConversationNotFoundError
from llama_stack_api.conversations import Conversations


def _build_app(impl):
    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    router = build_fastapi_router(Api.conversations, impl)
    assert router is not None
    app.include_router(router)
    return app


def test_get_conversation_maps_value_error_to_400():
    """ExceptionTranslatingRoute converts ValueError to HTTP 400."""
    impl = AsyncMock(spec=Conversations)
    impl.get_conversation.side_effect = ValueError("invalid value")

    client = TestClient(_build_app(impl), raise_server_exceptions=False)
    resp = client.get("/v1/conversations/conv_abc")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid value"


def test_get_conversation_maps_not_found_error():
    """ExceptionTranslatingRoute converts ConversationNotFoundError to HTTP 404."""
    impl = AsyncMock(spec=Conversations)
    impl.get_conversation.side_effect = ConversationNotFoundError("conv_missing")

    client = TestClient(_build_app(impl), raise_server_exceptions=False)
    resp = client.get("/v1/conversations/conv_missing")

    assert resp.status_code == 404
    assert "conv_missing" in resp.json()["detail"]


def test_unknown_exception_propagates_to_global_handler():
    """Unknown exceptions propagate past the route class to the global handler."""
    impl = AsyncMock(spec=Conversations)
    impl.get_conversation.side_effect = RuntimeError("something broke")

    client = TestClient(_build_app(impl), raise_server_exceptions=False)
    resp = client.get("/v1/conversations/conv_abc")

    assert resp.status_code == 500
    assert "error" in resp.json()


def test_consecutive_errors_keep_connection_alive():
    """Route-level translation prevents connection resets on repeated errors."""
    impl = AsyncMock(spec=Conversations)
    impl.get_conversation.side_effect = ValueError("bad request")

    client = TestClient(_build_app(impl), raise_server_exceptions=False)

    resp1 = client.get("/v1/conversations/conv_abc")
    assert resp1.status_code == 400

    resp2 = client.get("/v1/conversations/conv_abc")
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "bad request"


def test_delete_conversation_maps_value_error_to_400():
    """ExceptionTranslatingRoute converts ValueError on DELETE to HTTP 400."""
    impl = AsyncMock(spec=Conversations)
    impl.openai_delete_conversation.side_effect = ValueError("cannot delete")

    client = TestClient(_build_app(impl), raise_server_exceptions=False)
    resp = client.delete("/v1/conversations/conv_abc")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "cannot delete"
