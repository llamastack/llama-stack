# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import yaml

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.persistence_resolver import (
    resolve_inference_store_config,
    resolve_metadata_store_config,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import (
    PostgresSqlStoreConfig,
    SqliteSqlStoreConfig,
)


def test_starter_distribution_config_loads_and_resolves():
    """Integration: Actual starter config should parse and resolve all stores."""
    with open("llama_stack/distributions/starter/run.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackRunConfig(**config_dict)

    # Config should have persistence with default backend
    assert config.persistence is not None
    assert "default" in config.persistence.backends
    assert isinstance(config.persistence.backends["default"], SqliteSqlStoreConfig)

    # Stores should reference the default backend
    assert config.persistence.stores is not None
    assert config.persistence.stores.metadata.backend == "default"
    assert config.persistence.stores.inference.backend == "default"

    # Resolution should work
    metadata_store = resolve_metadata_store_config(config.persistence, "starter")
    assert isinstance(metadata_store, SqliteKVStoreConfig)

    sql_config, max_queue, num_writers = resolve_inference_store_config(config.persistence)
    assert isinstance(sql_config, SqliteSqlStoreConfig)
    assert max_queue > 0
    assert num_writers > 0


def test_postgres_demo_distribution_config_loads():
    """Integration: Postgres demo should use Postgres backend for all stores."""
    with open("llama_stack/distributions/postgres-demo/run.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackRunConfig(**config_dict)

    # Should have postgres backend
    assert config.persistence is not None
    assert "default" in config.persistence.backends
    assert isinstance(config.persistence.backends["default"], PostgresSqlStoreConfig)

    # Both stores use same postgres backend
    assert config.persistence.stores.metadata.backend == "default"
    assert config.persistence.stores.inference.backend == "default"

    # Resolution returns postgres config
    sql_config, _, _ = resolve_inference_store_config(config.persistence)
    assert isinstance(sql_config, PostgresSqlStoreConfig)
    assert sql_config.host == "${env.POSTGRES_HOST:=localhost}"
