# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore.config import KVStoreConfig, SqliteKVStoreConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ChromaVectorIOConfig(BaseModel):
    db_path: str
    kvstore: KVStoreConfig = Field(description="Config for KV store backend")

    @classmethod
    def sample_run_config(
        cls, __distro_dir__: str, db_path: str = "${env.CHROMADB_PATH}", **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "db_path": db_path,
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="chroma_inline_registry.db",
            ),
        }
