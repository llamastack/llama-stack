# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig, SqlStoreConfig


class LocalfsFilesImplConfig(BaseModel):
    storage_dir: str = Field(
        description="Directory to store uploaded files",
    )
    metadata_store: SqlStoreConfig = Field(
        description="SQL store configuration for file metadata",
    )
    ttl_secs: int = 365 * 24 * 60 * 60  # 1 year

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "storage_dir": "${env.FILES_STORAGE_DIR:=" + __distro_dir__ + "/files}",
            "metadata_store": SqliteSqlStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="files_metadata.db",
            ),
        }
