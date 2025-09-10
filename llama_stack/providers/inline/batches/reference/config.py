# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore.config import KVStoreConfig, SqliteKVStoreConfig


class ReferenceBatchesImplConfig(BaseModel):
    """Configuration for the Reference Batches implementation."""

    kvstore: KVStoreConfig = Field(
        description="Configuration for the key-value store backend.",
    )

    max_concurrent_batches: int = Field(
        default=1,
        description="Maximum number of concurrent batches to process simultaneously.",
        ge=1,
    )

    max_concurrent_requests_per_batch: int = Field(
        default=10,
        description="Maximum number of concurrent requests to process per batch.",
        ge=1,
    )

    # TODO: add a max requests per second rate limiter

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict:
        return {
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="batches.db",
            ),
        }
