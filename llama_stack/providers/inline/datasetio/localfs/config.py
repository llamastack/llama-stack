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


class LocalFSDatasetIOConfig(BaseModel):
    kvstore: KVStoreConfig = Field(default_factory=SqliteKVStoreConfig)

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="localfs_datasetio.db",
            )
        }
