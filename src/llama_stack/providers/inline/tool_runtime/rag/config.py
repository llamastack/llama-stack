# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import VectorStoresConfig


class RagToolRuntimeConfig(BaseModel):
    vector_stores_config: VectorStoresConfig = Field(
        default_factory=VectorStoresConfig,
        description="Configuration for vector store prompt templates and behavior",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}
