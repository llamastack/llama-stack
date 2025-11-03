# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import (
    Chunk,
    ChunkMetadata,
    InsertChunksRequest,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksRequest,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreContent,
    VectorStoreCreateRequest,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentsResponse,
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
from .vector_io_service import VectorIOService, VectorStoreTable

# Backward compatibility - export VectorIO as alias for VectorIOService
VectorIO = VectorIOService

__all__ = [
    "VectorIO",
    "VectorIOService",
    "VectorStoreTable",
    "Chunk",
    "ChunkMetadata",
    "QueryChunksResponse",
    "InsertChunksRequest",
    "QueryChunksRequest",
    "VectorStoreObject",
    "VectorStoreCreateRequest",
    "VectorStoreModifyRequest",
    "VectorStoreListResponse",
    "VectorStoreDeleteResponse",
    "VectorStoreSearchRequest",
    "VectorStoreSearchResponse",
    "VectorStoreSearchResponsePage",
    "VectorStoreContent",
    "VectorStoreFileCounts",
    "VectorStoreFileObject",
    "VectorStoreListFilesResponse",
    "VectorStoreFileContentsResponse",
    "VectorStoreFileDeleteResponse",
    "VectorStoreFileBatchObject",
    "VectorStoreFilesListInBatchResponse",
    "VectorStoreChunkingStrategy",
    "VectorStoreChunkingStrategyAuto",
    "VectorStoreChunkingStrategyStatic",
    "VectorStoreChunkingStrategyStaticConfig",
    "VectorStoreFileStatus",
    "VectorStoreFileLastError",
    "SearchRankingOptions",
    "OpenAICreateVectorStoreRequestWithExtraBody",
    "OpenAICreateVectorStoreFileBatchRequestWithExtraBody",
]
