# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from llama_stack.core.datatypes import (
    InferenceStoreReference,
    PersistenceConfig,
    StoreReference,
    StoresConfig,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import (
    PostgresSqlStoreConfig,
    SqliteSqlStoreConfig,
)


def test_backend_reference_validation_catches_missing_backend():
    """Critical: Catch user typos in backend references before runtime."""
    with pytest.raises(ValidationError, match="not defined in persistence.backends"):
        PersistenceConfig(
            backends={
                "default": SqliteSqlStoreConfig(db_path="/tmp/store.db"),
            },
            stores=StoresConfig(
                metadata=StoreReference(backend="typo_backend"),  # User typo
            ),
        )


def test_backend_reference_validation_accepts_valid_config():
    """Valid config should parse without errors."""
    config = PersistenceConfig(
        backends={
            "default": SqliteSqlStoreConfig(db_path="/tmp/store.db"),
        },
        stores=StoresConfig(
            metadata=StoreReference(backend="default"),
            inference=InferenceStoreReference(backend="default"),
        ),
    )
    assert config.stores.metadata.backend == "default"
    assert config.stores.inference.backend == "default"


def test_multiple_stores_can_share_same_backend():
    """Core use case: metadata and inference both use 'default' backend."""
    config = PersistenceConfig(
        backends={
            "default": SqliteSqlStoreConfig(db_path="/tmp/shared.db"),
        },
        stores=StoresConfig(
            metadata=StoreReference(backend="default", namespace="metadata"),
            inference=InferenceStoreReference(backend="default"),
            conversations=StoreReference(backend="default"),
        ),
    )
    # All reference the same backend
    assert config.stores.metadata.backend == "default"
    assert config.stores.inference.backend == "default"
    assert config.stores.conversations.backend == "default"


def test_mixed_backend_types_allowed():
    """Should support KVStore and SqlStore backends simultaneously."""
    config = PersistenceConfig(
        backends={
            "kvstore": SqliteKVStoreConfig(db_path="/tmp/kv.db"),
            "sqlstore": PostgresSqlStoreConfig(user="test", password="test", host="localhost", db="test"),
        },
        stores=StoresConfig(
            metadata=StoreReference(backend="kvstore"),
            inference=InferenceStoreReference(backend="sqlstore"),
        ),
    )
    assert isinstance(config.backends["kvstore"], SqliteKVStoreConfig)
    assert isinstance(config.backends["sqlstore"], PostgresSqlStoreConfig)
