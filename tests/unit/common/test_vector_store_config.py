# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.vector_store_config import VectorStoreConfig


def test_defaults(monkeypatch):
    # ensure env is clean to avoid flaky defaults
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", raising=False)
    config = VectorStoreConfig()
    assert config.default_embedding_model is None
    assert config.default_embedding_dimension is None


def test_env_loading(monkeypatch):
    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", "123")

    config = VectorStoreConfig()
    assert config.default_embedding_model == "test-model"
    assert config.default_embedding_dimension == 123

    # cleanup
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", raising=False)
