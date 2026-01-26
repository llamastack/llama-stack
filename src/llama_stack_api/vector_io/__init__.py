# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""VectorIO API module.

This module provides the VectorIO API for vector database operations.
Protocol definitions are in api.py, Pydantic models are in models.py,
and FastAPI routes are in fastapi_routes.py.
"""

# Re-export Protocol classes from api.py
from . import fastapi_routes
from .api import VectorIO, VectorStoreTable

# Re-export all Pydantic models from models.py
from .models import (
    Chunk,
    ChunkMetadata,
    EmbeddedChunk,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreContent,
    VectorStoreCreateRequest,
    VectorStoreDeleteResponse,
    VectorStoreFileAttributes,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileCounts,
    VectorStoreFileDeleteResponse,
    VectorStoreFileLastError,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreModifyRequest,
    VectorStoreObject,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)

__all__ = [
    # Protocol classes
    "VectorIO",
    "VectorStoreTable",
    # Pydantic models
    "Chunk",
    "ChunkMetadata",
    "EmbeddedChunk",
    "OpenAICreateVectorStoreFileBatchRequestWithExtraBody",
    "OpenAICreateVectorStoreRequestWithExtraBody",
    "QueryChunksResponse",
    "SearchRankingOptions",
    "VectorStoreChunkingStrategy",
    "VectorStoreChunkingStrategyAuto",
    "VectorStoreChunkingStrategyStatic",
    "VectorStoreChunkingStrategyStaticConfig",
    "VectorStoreContent",
    "VectorStoreCreateRequest",
    "VectorStoreDeleteResponse",
    "VectorStoreFileAttributes",
    "VectorStoreFileBatchObject",
    "VectorStoreFileContentResponse",
    "VectorStoreFileCounts",
    "VectorStoreFileDeleteResponse",
    "VectorStoreFileLastError",
    "VectorStoreFileObject",
    "VectorStoreFileStatus",
    "VectorStoreFilesListInBatchResponse",
    "VectorStoreListFilesResponse",
    "VectorStoreListResponse",
    "VectorStoreModifyRequest",
    "VectorStoreObject",
    "VectorStoreSearchRequest",
    "VectorStoreSearchResponse",
    "VectorStoreSearchResponsePage",
    "fastapi_routes",
]
