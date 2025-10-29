# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ElasticsearchVectorIOConfig(BaseModel):
    elasticsearch_api_key: str | None = Field(description="The API key for the Elasticsearch instance", default=None)
    elasticsearch_url: str | None = Field(description="The URL of the Elasticsearch instance", default="localhost:9200")
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend (SQLite only for now)", default=None
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "elasticsearch_api_key": None,
            "elasticsearch_url": "${env.ELASTICSEARCH_URL:=localhost:9200}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::elasticsearch",
            ).model_dump(exclude_none=True),
        }