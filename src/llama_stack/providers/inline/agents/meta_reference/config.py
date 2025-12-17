# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.core.storage.datatypes import KVStoreReference, ResponsesStoreReference


class AgentPersistenceConfig(BaseModel):
    """Nested persistence configuration for agents."""

    agent_state: KVStoreReference
    responses: ResponsesStoreReference


class MetaReferenceAgentsImplConfig(BaseModel):
    persistence: AgentPersistenceConfig
    vector_stores_config: VectorStoresConfig | None = Field(
        default=None,
        description="Configuration for vector store prompt templates and behavior",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "persistence": {
                "agent_state": KVStoreReference(
                    backend="kv_default",
                    namespace="agents",
                ).model_dump(exclude_none=True),
                "responses": ResponsesStoreReference(
                    backend="sql_default",
                    table_name="responses",
                ).model_dump(exclude_none=True),
            }
        }
