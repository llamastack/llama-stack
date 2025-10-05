# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.core.datatypes import (
    InferenceStoreReference,
    PersistenceConfig,
    StoreReference,
    StoresConfig,
)
from llama_stack.core.persistence_resolver import (
    resolve_backend,
    resolve_inference_store_config,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig


def test_resolver_applies_namespace_to_kvstore():
    """Critical: Namespace overlay must work for KVStore isolation."""
    persistence = PersistenceConfig(
        backends={
            "default": SqliteKVStoreConfig(db_path="/tmp/store.db"),
        },
        stores=StoresConfig(
            metadata=StoreReference(backend="default", namespace="meta"),
        ),
    )

    resolved = resolve_backend(
        persistence=persistence,
        store_ref=persistence.stores.metadata,
        default_factory=lambda: SqliteKVStoreConfig(db_path="/tmp/default.db"),
        store_name="metadata",
    )

    # Backend config cloned with namespace applied
    assert resolved.db_path == "/tmp/store.db"
    assert resolved.namespace == "meta"


def test_resolver_does_not_apply_namespace_to_sqlstore():
    """SqlStore backends should not get namespace field."""
    persistence = PersistenceConfig(
        backends={
            "default": SqliteSqlStoreConfig(db_path="/tmp/store.db"),
        },
        stores=StoresConfig(
            inference=InferenceStoreReference(backend="default"),
        ),
    )

    sql_config, _, _ = resolve_inference_store_config(persistence)

    # SqlStore returned as-is, no namespace attribute
    assert sql_config.db_path == "/tmp/store.db"
    assert not hasattr(sql_config, "namespace")


def test_resolver_rejects_kvstore_for_inference():
    """Type safety: inference requires SqlStore, should fail on KVStore."""
    persistence = PersistenceConfig(
        backends={
            "default": SqliteKVStoreConfig(db_path="/tmp/kv.db"),  # Wrong type
        },
        stores=StoresConfig(
            inference=InferenceStoreReference(backend="default"),
        ),
    )

    with pytest.raises(ValueError, match="requires SqlStore backend"):
        resolve_inference_store_config(persistence)


def test_resolver_preserves_queue_params():
    """Inference store should preserve queue tuning parameters."""
    persistence = PersistenceConfig(
        backends={
            "default": SqliteSqlStoreConfig(db_path="/tmp/store.db"),
        },
        stores=StoresConfig(
            inference=InferenceStoreReference(
                backend="default",
                max_write_queue_size=5000,
                num_writers=2,
            ),
        ),
    )

    _, max_queue, num_writers = resolve_inference_store_config(persistence)

    assert max_queue == 5000
    assert num_writers == 2


def test_resolver_uses_defaults_when_no_persistence_config():
    """Graceful fallback when persistence not configured."""
    sql_config, max_queue, num_writers = resolve_inference_store_config(None)

    # Should return sensible defaults
    assert isinstance(sql_config, SqliteSqlStoreConfig)
    assert max_queue == 10000
    assert num_writers == 4
