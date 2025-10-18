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
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)


def _base_run_config(**overrides):
    storage = overrides.pop(
        "storage",
        StorageConfig(
            backends={
                "kv_default": SqliteKVStoreConfig(db_path="/tmp/kv.db"),
                "sql_default": SqliteSqlStoreConfig(db_path="/tmp/sql.db"),
            }
        ),
    )
    return StackRunConfig(
        version=LLAMA_STACK_RUN_CONFIG_VERSION,
        image_name="test-distro",
        apis=[],
        providers={},
        storage=storage,
        metadata_store=overrides.pop(
            "metadata_store",
            KVStoreReference(backend="kv_default", namespace="registry"),
        ),
        inference_store=overrides.pop(
            "inference_store",
            InferenceStoreReference(backend="sql_default", table_name="inference"),
        ),
        conversations_store=overrides.pop(
            "conversations_store",
            SqlStoreReference(backend="sql_default", table_name="conversations"),
        ),
        **overrides,
    )


def test_references_require_known_backend():
    with pytest.raises(ValidationError, match="unknown backend 'missing'"):
        _base_run_config(metadata_store=KVStoreReference(backend="missing", namespace="registry"))


def test_references_must_match_backend_family():
    with pytest.raises(ValidationError, match="kv_.* is required"):
        _base_run_config(metadata_store=KVStoreReference(backend="sql_default", namespace="registry"))

    with pytest.raises(ValidationError, match="sql_.* is required"):
        _base_run_config(
            inference_store=InferenceStoreReference(backend="kv_default", table_name="inference"),
        )


def test_valid_configuration_passes_validation():
    config = _base_run_config()
    assert config.metadata_store.backend == "kv_default"
    assert config.inference_store.backend == "sql_default"
    assert config.conversations_store.backend == "sql_default"
