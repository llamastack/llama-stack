# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.models import ModelType
from llama_stack.core.routers.vector_io import VectorIORouter

pytestmark = pytest.mark.asyncio


class _DummyModel:
    def __init__(self, identifier: str, dim: int):
        self.identifier = identifier
        self.model_type = ModelType.embedding
        self.metadata = {"embedding_dimension": dim}


class _DummyRoutingTable:
    """Just a fake routing table for testing."""

    def __init__(self):
        self._models = [
            _DummyModel("first-model", 123),
            _DummyModel("second-model", 512),
        ]

    async def get_all_with_type(self, _type: str):
        # just return embedding models for tests
        return self._models

    # VectorIORouter needs these but we don't use them in tests
    async def register_vector_db(self, *_args, **_kwargs):
        raise NotImplementedError

    async def get_provider_impl(self, *_args, **_kwargs):
        raise NotImplementedError


async def test_global_default_used(monkeypatch):
    """Should use env var defaults when no explicit model given."""

    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", "env-default-model")
    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", "256")

    router = VectorIORouter(routing_table=_DummyRoutingTable())

    model, dim = await router._resolve_embedding_model(None)
    assert model == "env-default-model"
    assert dim == 256

    # cleanup
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", raising=False)


async def test_explicit_override(monkeypatch):
    """Explicit model should win over env defaults."""

    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", "env-default-model")

    router = VectorIORouter(routing_table=_DummyRoutingTable())

    model, dim = await router._resolve_embedding_model("first-model")
    assert model == "first-model"
    assert dim == 123

    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", raising=False)


async def test_fallback_to_default():
    """Should fallback to all-MiniLM-L6-v2 when no defaults set."""

    router = VectorIORouter(routing_table=_DummyRoutingTable())

    model, dim = await router._resolve_embedding_model(None)
    assert model == "all-MiniLM-L6-v2"
    assert dim == 384
