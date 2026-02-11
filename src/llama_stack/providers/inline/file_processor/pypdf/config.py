# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class PyPDFFileProcessorConfig(BaseModel):
    """Configuration for PyPDF file processor."""

    # Chunking configuration for RAG/vector store integration
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

    # PDF-specific options
    extract_metadata: bool = Field(default=True, description="Whether to extract PDF metadata (title, author, etc.)")

    # Text processing options
    clean_text: bool = Field(
        default=True, description="Whether to clean extracted text (remove extra whitespace, normalize line breaks)"
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {}
