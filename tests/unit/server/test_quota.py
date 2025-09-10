# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from llama_stack.core.datatypes import QuotaConfig, QuotaPeriod
from llama_stack.core.server.quota import QuotaMiddleware
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


class InjectClientIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects 'authenticated_client_id' to mimic AuthenticationMiddleware.
    """

    def __init__(self, app, client_id="client1"):
        super().__init__(app)
        self.client_id = client_id

    async def dispatch(self, request: Request, call_next):
        request.scope["authenticated_client_id"] = self.client_id
        return await call_next(request)


def build_quota_config(db_path) -> QuotaConfig:
    return QuotaConfig(
        kvstore=SqliteKVStoreConfig(db_path=str(db_path)),
        anonymous_max_requests=1,
        authenticated_max_requests=2,
        period=QuotaPeriod.DAY,
    )


@pytest.fixture
def auth_app(tmp_path, request):
    """
    FastAPI app with InjectClientIDMiddleware and QuotaMiddleware for authenticated testing.
    Each test gets its own DB file.
    """
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    db_path = tmp_path / f"quota_{request.node.name}.db"
    quota = build_quota_config(db_path)

    app = InjectClientIDMiddleware(
        QuotaMiddleware(
            inner_app,
            kv_config=quota.kvstore,
            anonymous_max_requests=quota.anonymous_max_requests,
            authenticated_max_requests=quota.authenticated_max_requests,
            window_seconds=86400,
        ),
        client_id=f"client_{request.node.name}",
    )
    return app


def test_authenticated_quota_allows_up_to_limit(auth_app):
    client = TestClient(auth_app)
    assert client.get("/test").status_code == 200
    assert client.get("/test").status_code == 200


def test_authenticated_quota_blocks_after_limit(auth_app):
    client = TestClient(auth_app)
    client.get("/test")
    client.get("/test")
    resp = client.get("/test")
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "Quota exceeded"


def test_anonymous_quota_allows_up_to_limit(tmp_path, request):
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    db_path = tmp_path / f"quota_anon_{request.node.name}.db"
    quota = build_quota_config(db_path)

    app = QuotaMiddleware(
        inner_app,
        kv_config=quota.kvstore,
        anonymous_max_requests=quota.anonymous_max_requests,
        authenticated_max_requests=quota.authenticated_max_requests,
        window_seconds=86400,
    )

    client = TestClient(app)
    assert client.get("/test").status_code == 200


def test_anonymous_quota_blocks_after_limit(tmp_path, request):
    inner_app = FastAPI()

    @inner_app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    db_path = tmp_path / f"quota_anon_{request.node.name}.db"
    quota = build_quota_config(db_path)

    app = QuotaMiddleware(
        inner_app,
        kv_config=quota.kvstore,
        anonymous_max_requests=quota.anonymous_max_requests,
        authenticated_max_requests=quota.authenticated_max_requests,
        window_seconds=86400,
    )

    client = TestClient(app)
    client.get("/test")
    resp = client.get("/test")
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "Quota exceeded"
