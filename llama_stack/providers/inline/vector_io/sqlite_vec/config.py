# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)


class SQLiteVectorIOConfig(BaseModel):
    db_path: str = Field(description="Path to the SQLite database file")
    kvstore: KVStoreConfig = Field(description="Config for KV store backend (SQLite only for now)")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + "sqlite_vec.db",
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="sqlite_vec_registry.db",
            ),
        }
