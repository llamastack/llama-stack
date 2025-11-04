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
class OpenAIVectorIOConfig(BaseModel):
    api_key: str | None = Field(
        None,
        description="OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.",
    )
    persistence: KVStoreReference = Field(
        description="KVStore reference for persisting vector store metadata.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${OPENAI_API_KEY}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::openai",
            ).model_dump(exclude_none=True),
        }
