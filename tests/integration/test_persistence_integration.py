# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import yaml

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.storage.datatypes import (
    PostgresSqlStoreConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
)


def test_starter_distribution_config_loads_and_resolves():
    """Integration: Actual starter config should parse and have correct storage structure."""
    with open("llama_stack/distributions/starter/run.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackRunConfig(**config_dict)

    # Config should have storage with default backend
    assert config.storage is not None
    assert "default" in config.storage.backends
    assert isinstance(config.storage.backends["default"], SqliteSqlStoreConfig)

    # Stores should reference the default backend
    assert config.storage.metadata is not None
    assert config.storage.metadata.backend == "default"
    assert config.storage.metadata.namespace is not None

    assert config.storage.inference is not None
    assert config.storage.inference.backend == "default"
    assert config.storage.inference.table_name is not None
    assert config.storage.inference.max_write_queue_size > 0
    assert config.storage.inference.num_writers > 0


def test_postgres_demo_distribution_config_loads():
    """Integration: Postgres demo should use Postgres backend for all stores."""
    with open("llama_stack/distributions/postgres-demo/run.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackRunConfig(**config_dict)

    # Should have postgres backend
    assert config.storage is not None
    assert "default" in config.storage.backends
    assert isinstance(config.storage.backends["default"], PostgresSqlStoreConfig)

    # Both stores use same postgres backend
    assert config.storage.metadata is not None
    assert config.storage.metadata.backend == "default"

    assert config.storage.inference is not None
    assert config.storage.inference.backend == "default"

    # Backend config should be Postgres
    postgres_backend = config.storage.backends["default"]
    assert isinstance(postgres_backend, PostgresSqlStoreConfig)
    assert postgres_backend.host == "${env.POSTGRES_HOST:=localhost}"
