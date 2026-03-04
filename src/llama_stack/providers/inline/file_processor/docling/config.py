# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class DoclingFileProcessorConfig(BaseModel):
    """Configuration for Docling file processor."""

    default_chunk_size_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["max_chunk_size_tokens"].default,
        ge=100,
        le=4096,
        description="Default chunk size in tokens when chunking_strategy type is 'auto'",
    )
    default_chunk_overlap_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["chunk_overlap_tokens"].default,
        ge=0,
        le=2048,
        description="Default chunk overlap in tokens when chunking_strategy type is 'auto'",
    )

    max_file_size_bytes: int = Field(
        default=100 * 1024 * 1024,
        ge=1,
        description="Maximum file size in bytes for uploaded files (default 100MB)",
    )

    max_page_count: int = Field(
        default=100,
        ge=1,
        description="Maximum number of pages to process (docling safeguard for CPU cost)",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {}
