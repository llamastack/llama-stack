# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore import KVStoreConfig
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig, SqlStoreConfig


class MetaReferenceAgentsImplConfig(BaseModel):
    persistence_store: KVStoreConfig = Field(default_factory=SqliteKVStoreConfig)
    responses_store: SqlStoreConfig = Field(default_factory=SqliteSqlStoreConfig)

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "persistence_store": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="agents_store.db",
            ),
            "responses_store": SqliteSqlStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="responses_store.db",
            ),
        }
