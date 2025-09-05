# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

"""Vector store global config stuff.

Basically just holds default embedding model settings so we don't have to
pass them around everywhere. Router picks these up when client doesn't specify.
"""

import os

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["VectorStoreConfig"]


class VectorStoreConfig(BaseModel):
    """Default embedding model config that gets picked up from env vars."""

    default_embedding_model: str | None = Field(
        default_factory=lambda: os.getenv("LLAMA_STACK_DEFAULT_EMBEDDING_MODEL")
    )
    # dimension from env - fallback to None if not set or invalid
    default_embedding_dimension: int | None = Field(
        default_factory=lambda: int(os.getenv("LLAMA_STACK_DEFAULT_EMBEDDING_DIMENSION", 0)) or None, ge=1
    )

    model_config = ConfigDict(frozen=True)
