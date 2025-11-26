# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for storage backend/reference validation."""

import pytest
from pydantic import ValidationError

from llama_stack.core.datatypes import (
    LLAMA_STACK_RUN_CONFIG_VERSION,
    StackRunConfig,
)
from llama_stack.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)


def _base_run_config(**overrides):
    metadata_reference = overrides.pop(
        "metadata_reference",
        KVStoreReference(backend="kv_default", namespace="registry"),
    )
    inference_reference = overrides.pop(
        "inference_reference",
        InferenceStoreReference(backend="sql_default", table_name="inference"),
    )
    conversations_reference = overrides.pop(
        "conversations_reference",
        SqlStoreReference(backend="sql_default", table_name="conversations"),
    )
    storage = overrides.pop(
        "storage",
        StorageConfig(
            backends={
                "kv_default": SqliteKVStoreConfig(db_path="/tmp/kv.db"),
                "sql_default": SqliteSqlStoreConfig(db_path="/tmp/sql.db"),
            },
            stores=ServerStoresConfig(
                metadata=metadata_reference,
                inference=inference_reference,
                conversations=conversations_reference,
            ),
        ),
    )
    return StackRunConfig(
        version=LLAMA_STACK_RUN_CONFIG_VERSION,
        image_name="test-distro",
        apis=[],
        providers={},
        storage=storage,
        **overrides,
    )


def test_references_require_known_backend():
    with pytest.raises(ValidationError, match="unknown backend 'missing'"):
        _base_run_config(metadata_reference=KVStoreReference(backend="missing", namespace="registry"))


def test_references_must_match_backend_family():
    with pytest.raises(ValidationError, match="kv_.* is required"):
        _base_run_config(metadata_reference=KVStoreReference(backend="sql_default", namespace="registry"))

    with pytest.raises(ValidationError, match="sql_.* is required"):
        _base_run_config(
            inference_reference=InferenceStoreReference(backend="kv_default", table_name="inference"),
        )


def test_valid_configuration_passes_validation():
    config = _base_run_config()
    stores = config.storage.stores
    assert stores.metadata is not None and stores.metadata.backend == "kv_default"
    assert stores.inference is not None and stores.inference.backend == "sql_default"
    assert stores.conversations is not None and stores.conversations.backend == "sql_default"
