# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference


class ReferencePromptsConfig(BaseModel):
    """Configuration for the built-in reference prompt service.

    This provider stores prompts in the configured KVStore (SQLite, PostgreSQL, etc.)
    as specified in the run configuration.
    """

    prompts_store: KVStoreReference = Field(description="KVStore reference for prompts storage")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "prompts_store": KVStoreReference(
                backend="kv_default",
                namespace="prompts",
            ).model_dump(exclude_none=True),
        }
