# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.core.storage.datatypes import KVStoreReference, SqlStoreReference


class AgentPersistenceConfig(BaseModel):
    """Nested persistence configuration for agents."""

    agent_state: KVStoreReference
    responses: SqlStoreReference


class MetaReferenceAgentsImplConfig(BaseModel):
    persistence: AgentPersistenceConfig

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "persistence": {
                "agent_state": KVStoreReference(
                    namespace="agents",
                ).model_dump(exclude_none=True),
                "responses": SqlStoreReference(
                    table_name="responses",
                ).model_dump(exclude_none=True),
            }
        }
