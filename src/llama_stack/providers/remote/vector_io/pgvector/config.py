# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack_api import json_schema_type


@json_schema_type
class PGVectorVectorIOConfig(BaseModel):
    host: str | None = Field(default="localhost")
    port: int | None = Field(default=5432)
    db: str | None = Field(default="postgres")
    user: str | None = Field(default="postgres")
    password: str | None = Field(default="mysecretpassword")
    distance_metric: Literal["COSINE", "L2", "L1", "INNER_PRODUCT"] | None = Field(
        default="COSINE", description="PGVector distance metric used for vector search in PGVectorIndex"
    )
    hnsw_m: int | None = Field(
        gt=0,
        default=16,
        description="PGVector's HNSW index parameter - maximum number of edges each vertex has to its neighboring vertices in the graph",
    )
    hnsw_ef_construction: int | None = Field(
        gt=0,
        default=64,
        description="PGVector's HNSW index parameter - size of the dynamic candidate list used for graph construction",
    )
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend (SQLite only for now)", default=None
    )

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        host: str = "${env.PGVECTOR_HOST:=localhost}",
        port: int = "${env.PGVECTOR_PORT:=5432}",
        db: str = "${env.PGVECTOR_DB}",
        user: str = "${env.PGVECTOR_USER}",
        password: str = "${env.PGVECTOR_PASSWORD}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "host": host,
            "port": port,
            "db": db,
            "user": user,
            "password": password,
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::pgvector",
            ).model_dump(exclude_none=True),
        }
