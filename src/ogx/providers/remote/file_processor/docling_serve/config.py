# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class DoclingServeFileProcessorConfig(BaseModel):
    """Configuration for remote Docling Serve file processor."""

    base_url: str = Field(
        default="http://localhost:5001/v1",
        description="Base URL of the Docling Serve instance",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for authenticating with Docling Serve (optional, required if server has DOCLING_SERVE_API_KEY set)",
    )
    default_chunk_size_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["max_chunk_size_tokens"].default,
        ge=100,
        le=4096,
        description="Default chunk size in tokens when chunking_strategy type is 'auto'",
    )
    max_concurrency: int = Field(
        default=2,
        ge=1,
        le=64,
        description="Maximum number of concurrent requests to Docling Serve. "
        "Excess requests queue in FIFO order. Lower values protect "
        "the server from resource exhaustion on large or image-heavy files.",
    )
    max_queue_size: int = Field(
        default=0,
        ge=0,
        description="Maximum number of requests waiting in the queue. "
        "0 means unlimited. When the queue is full, new requests "
        "are rejected immediately with an error.",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "base_url": "${env.DOCLING_SERVE_URL:=http://localhost:5001/v1}",
            "api_key": "${env.DOCLING_SERVE_API_KEY:=}",
            "max_concurrency": "${env.DOCLING_SERVE_MAX_CONCURRENCY:=2}",
        }
