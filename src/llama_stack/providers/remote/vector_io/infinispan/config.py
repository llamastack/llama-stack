# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack_api import json_schema_type


@json_schema_type
class InfinispanVectorIOConfig(BaseModel):
    url: str = Field(
        default="http://localhost:11222",
        description="Infinispan server URL (e.g., http://localhost:11222)"
    )
    username: str | None = Field(
        default=None,
        description="Authentication username"
    )
    password: str | None = Field(
        default=None,
        description="Authentication password"
    )
    use_https: bool = Field(
        default=False,
        description="Enable HTTPS/TLS connection"
    )
    auth_mechanism: str = Field(
        default="digest",
        description="Authentication mechanism: 'digest' or 'basic'"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates for HTTPS connections"
    )
    persistence: KVStoreReference = Field(
        description="Config for KV store backend"
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "url": "${env.INFINISPAN_URL:=http://localhost:11222}",
            "username": "${env.INFINISPAN_USERNAME:=admin}",
            "password": "${env.INFINISPAN_PASSWORD:=password}",
            "use_https": False,
            "auth_mechanism": "digest",
            "verify_ssl": True,
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::infinispan",
            ).model_dump(exclude_none=True),
        }
