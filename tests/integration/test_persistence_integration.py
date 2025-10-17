# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import yaml

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.storage.datatypes import (
    PostgresKVStoreConfig,
    PostgresSqlStoreConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
)


def test_starter_distribution_config_loads_and_resolves():
    """Integration: Actual starter config should parse and have correct storage structure."""
    with open("llama_stack/distributions/starter/run.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackRunConfig(**config_dict)

    # Config should have named backends and explicit store references
    assert config.storage is not None
    assert "kv_default" in config.storage.backends
    assert "sql_default" in config.storage.backends
    assert isinstance(config.storage.backends["kv_default"], SqliteKVStoreConfig)
    assert isinstance(config.storage.backends["sql_default"], SqliteSqlStoreConfig)

    assert config.metadata_store is not None
    assert config.metadata_store.backend == "kv_default"
    assert config.metadata_store.namespace == "registry"

    assert config.inference_store is not None
    assert config.inference_store.backend == "sql_default"
    assert config.inference_store.table_name == "inference_store"
    assert config.inference_store.max_write_queue_size > 0
    assert config.inference_store.num_writers > 0

    assert config.conversations_store is not None
    assert config.conversations_store.backend == "sql_default"
    assert config.conversations_store.table_name == "openai_conversations"


def test_postgres_demo_distribution_config_loads():
    """Integration: Postgres demo should use Postgres backend for all stores."""
    with open("llama_stack/distributions/postgres-demo/run.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackRunConfig(**config_dict)

    # Should have postgres backend
    assert config.storage is not None
    assert "kv_default" in config.storage.backends
    assert "sql_default" in config.storage.backends
    postgres_backend = config.storage.backends["sql_default"]
    assert isinstance(postgres_backend, PostgresSqlStoreConfig)
    assert postgres_backend.host == "${env.POSTGRES_HOST:=localhost}"

    kv_backend = config.storage.backends["kv_default"]
    assert isinstance(kv_backend, PostgresKVStoreConfig)

    # Stores target the Postgres backends explicitly
    assert config.metadata_store.backend == "kv_default"
    assert config.inference_store.backend == "sql_default"
