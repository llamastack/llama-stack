# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import os
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from fastapi import APIRouter, FastAPI
from starlette.testclient import TestClient

from llama_stack.core.access_control.access_control import default_policy
from llama_stack.core.request_headers import get_authenticated_user
from llama_stack.core.server.server import ProviderDataMiddleware
from llama_stack.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from llama_stack.core.storage.sqlstore.sqlstore import SqliteSqlStoreConfig
from llama_stack.core.testing_context import get_test_context
from llama_stack_api.internal.sqlstore import ColumnType


@pytest.fixture
def app_with_middleware():
    """Create a minimal FastAPI app with ProviderDataMiddleware."""
    app = FastAPI()

    router = APIRouter()

    @router.get("/test-context")
    def get_current_test_context():
        return {"test_id": get_test_context()}

    app.include_router(router)
    app.add_middleware(ProviderDataMiddleware)

    return app


@pytest.fixture
def test_mode_env(monkeypatch):
    """Set environment variables required for test context extraction."""
    monkeypatch.setenv("LLAMA_STACK_TEST_INFERENCE_MODE", "replay")
    monkeypatch.setenv("LLAMA_STACK_TEST_STACK_CONFIG_TYPE", "server")


def test_middleware_returns_none_without_header(app_with_middleware, test_mode_env):
    """Without the provider data header, test context should be None."""
    client = TestClient(app_with_middleware)
    response = client.get("/test-context")

    assert response.status_code == 200
    assert response.json()["test_id"] is None


def test_middleware_extracts_test_id_from_header(app_with_middleware, test_mode_env):
    """With the provider data header containing __test_id, it should be extracted."""
    client = TestClient(app_with_middleware)

    provider_data = json.dumps({"__test_id": "test-abc-123"})
    response = client.get(
        "/test-context",
        headers={"X-LlamaStack-Provider-Data": provider_data},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] == "test-abc-123"


def test_middleware_handles_empty_provider_data(app_with_middleware, test_mode_env):
    """Empty provider data should result in None test context."""
    client = TestClient(app_with_middleware)

    response = client.get(
        "/test-context",
        headers={"X-LlamaStack-Provider-Data": "{}"},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] is None


def test_middleware_handles_invalid_json(app_with_middleware, test_mode_env):
    """Invalid JSON in header should not crash, test context should be None."""
    client = TestClient(app_with_middleware)

    response = client.get(
        "/test-context",
        headers={"X-LlamaStack-Provider-Data": "not-valid-json"},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] is None


def test_middleware_noop_without_test_mode(app_with_middleware):
    """Without test mode env vars, middleware should not extract test context."""
    # Ensure env vars are not set
    os.environ.pop("LLAMA_STACK_TEST_INFERENCE_MODE", None)
    os.environ.pop("LLAMA_STACK_TEST_STACK_CONFIG_TYPE", None)

    client = TestClient(app_with_middleware)

    provider_data = json.dumps({"__test_id": "test-abc-123"})
    response = client.get(
        "/test-context",
        headers={"X-LlamaStack-Provider-Data": provider_data},
    )

    assert response.status_code == 200
    assert response.json()["test_id"] is None


@pytest.fixture
def app_with_authorized_store():
    with TemporaryDirectory() as tmp_dir:
        base_store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/reserved-key.db"))
        auth_store = AuthorizedSqlStore(base_store, default_policy())
        asyncio.run(
            auth_store.create_table(
                table="docs",
                schema={
                    "id": ColumnType.STRING,
                    "title": ColumnType.STRING,
                },
            )
        )

        app = FastAPI()

        @app.get("/debug/auth-user")
        async def debug_auth_user():
            user = get_authenticated_user()
            return {
                "user_type": type(user).__name__ if user is not None else None,
                "user": user,
            }

        @app.post("/docs")
        async def create_doc():
            await auth_store.insert(
                "docs",
                {"id": str(uuid4()), "title": "created-via-http"},
            )
            return {"ok": True}

        app.add_middleware(ProviderDataMiddleware)
        yield app


def test_reserved___authenticated_user_is_ignored_by_provider_data_middleware(app_with_authorized_store):
    client = TestClient(app_with_authorized_store)
    payload = json.dumps({"__authenticated_user": {"principal": "attacker", "attributes": {"roles": ["admin"]}}})

    response = client.get("/debug/auth-user", headers={"X-LlamaStack-Provider-Data": payload})

    assert response.status_code == 200
    body = response.json()
    assert body == {"user_type": None, "user": None}


def test_reserved___authenticated_user_does_not_trigger_500_during_store_insert(app_with_authorized_store):
    payload = json.dumps({"__authenticated_user": {"principal": "attacker", "attributes": {"roles": ["admin"]}}})

    baseline_client = TestClient(app_with_authorized_store)
    baseline = baseline_client.post("/docs")
    assert baseline.status_code == 200
    assert baseline.json() == {"ok": True}

    attack_client = TestClient(app_with_authorized_store)
    attacked = attack_client.post("/docs", headers={"X-LlamaStack-Provider-Data": payload})
    assert attacked.status_code == 200
    assert attacked.json() == {"ok": True}
