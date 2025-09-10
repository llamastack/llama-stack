# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.core.store.registry import CachedDiskDistributionRegistry, DiskDistributionRegistry
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.kvstore.sqlite import SqliteKVStoreImpl


@pytest.fixture(scope="function")
async def sqlite_kvstore(tmp_path):
    db_path = tmp_path / "test_kv.db"
    kvstore_config = SqliteKVStoreConfig(db_path=db_path.as_posix())
    kvstore = SqliteKVStoreImpl(kvstore_config)
    await kvstore.initialize()
    yield kvstore


@pytest.fixture(scope="function")
async def disk_dist_registry(sqlite_kvstore):
    registry = DiskDistributionRegistry(sqlite_kvstore)
    await registry.initialize()
    yield registry


@pytest.fixture(scope="function")
async def cached_disk_dist_registry(sqlite_kvstore):
    registry = CachedDiskDistributionRegistry(sqlite_kvstore)
    await registry.initialize()
    yield registry
