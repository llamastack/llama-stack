# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class QdrantVectorIOConfig(BaseModel):
    location: str | None = None
    url: str | None = None
    port: int | None = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    https: bool | None = None
    api_key: str | None = None
    prefix: str | None = None
    timeout: int | None = None
    host: str | None = None
    persistence: KVStoreReference

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${env.QDRANT_API_KEY:=}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::qdrant_remote",
            ).model_dump(exclude_none=True),
        }
