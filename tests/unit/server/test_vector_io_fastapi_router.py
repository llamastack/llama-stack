# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llama_stack.core.server.fastapi_router_registry import build_fastapi_router
from llama_stack_api import VectorStoreListResponse, VectorStoreSearchResponsePage
from llama_stack_api.datatypes import Api


def test_vector_io_router_list_vector_stores() -> None:
    impl = AsyncMock()
    impl.openai_list_vector_stores = AsyncMock(return_value=VectorStoreListResponse(data=[]))

    router = build_fastapi_router(Api.vector_io, impl)
    assert router is not None

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/v1/vector_stores")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}

    impl.openai_list_vector_stores.assert_awaited_once()


def test_vector_io_router_search_vector_store_passes_body_fields() -> None:
    impl = AsyncMock()
    impl.openai_search_vector_store = AsyncMock(
        return_value=VectorStoreSearchResponsePage(search_query=["hello"], data=[])
    )

    router = build_fastapi_router(Api.vector_io, impl)
    assert router is not None

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/v1/vector_stores/vs_123/search",
        json={"query": "hello", "rewrite_query": True},
    )

    assert response.status_code == 200
    assert response.json()["object"] == "vector_store.search_results.page"

    impl.openai_search_vector_store.assert_awaited_once()
    _, kwargs = impl.openai_search_vector_store.call_args
    assert kwargs["vector_store_id"] == "vs_123"
    assert kwargs["query"] == "hello"
    assert kwargs["rewrite_query"] is True
    assert kwargs["search_mode"] == "vector"
