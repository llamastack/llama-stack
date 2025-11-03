# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.inference import InterleavedContent
from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class ChunkMetadata(BaseModel):
    """
    `ChunkMetadata` is backend metadata for a `Chunk` that is used to store additional information about the chunk that
        will not be used in the context during inference, but is required for backend functionality. The `ChunkMetadata`
        is set during chunk creation in `MemoryToolRuntimeImpl().insert()`and is not expected to change after.
        Use `Chunk.metadata` for metadata that will be used in the context during inference.
    """

    chunk_id: str | None = None
    document_id: str | None = None
    source: str | None = None
    created_timestamp: int | None = None
    updated_timestamp: int | None = None
    chunk_window: str | None = None
    chunk_tokenizer: str | None = None
    chunk_embedding_model: str | None = None
    chunk_embedding_dimension: int | None = None
    content_token_count: int | None = None
    metadata_token_count: int | None = None


@json_schema_type
class Chunk(BaseModel):
    """A chunk of content that can be inserted into a vector database."""

    content: InterleavedContent
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    chunk_metadata: ChunkMetadata | None = None

    @property
    def document_id(self) -> str | None:
        """Returns the document_id from either metadata or chunk_metadata, with metadata taking precedence."""
        # Check metadata first (takes precedence)
        doc_id = self.metadata.get("document_id")
        if doc_id is not None:
            if not isinstance(doc_id, str):
                raise TypeError(f"metadata['document_id'] must be a string, got {type(doc_id).__name__}: {doc_id!r}")
            return doc_id

        # Fall back to chunk_metadata if available (Pydantic ensures type safety)
        if self.chunk_metadata is not None:
            return self.chunk_metadata.document_id

        return None


@json_schema_type
class QueryChunksResponse(BaseModel):
    """Response from querying chunks in a vector database."""

    chunks: list[Chunk]
    scores: list[float]


@json_schema_type
class VectorStoreFileCounts(BaseModel):
    """File processing status counts for a vector store."""

    completed: int
    cancelled: int
    failed: int
    in_progress: int
    total: int


# TODO: rename this as OpenAIVectorStore
@json_schema_type
class VectorStoreObject(BaseModel):
    """OpenAI Vector Store object."""

    id: str
    object: str = "vector_store"
    created_at: int
    name: str | None = None
    usage_bytes: int = 0
    file_counts: VectorStoreFileCounts
    status: str = "completed"
    expires_after: dict[str, Any] | None = None
    expires_at: int | None = None
    last_active_at: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class VectorStoreCreateRequest(BaseModel):
    """Request to create a vector store."""

    name: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    expires_after: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class VectorStoreModifyRequest(BaseModel):
    """Request to modify a vector store."""

    name: str | None = None
    expires_after: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreListResponse(BaseModel):
    """Response from listing vector stores."""

    object: str = "list"
    data: list[VectorStoreObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


@json_schema_type
class VectorStoreSearchRequest(BaseModel):
    """Request to search a vector store."""

    query: str | list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int = 10
    ranking_options: dict[str, Any] | None = None
    rewrite_query: bool = False


@json_schema_type
class VectorStoreContent(BaseModel):
    """Content item from a vector store file or search result."""

    type: Literal["text"]
    text: str


@json_schema_type
class VectorStoreSearchResponse(BaseModel):
    """Response from searching a vector store."""

    file_id: str
    filename: str
    score: float
    attributes: dict[str, str | float | bool] | None = None
    content: list[VectorStoreContent]


@json_schema_type
class VectorStoreSearchResponsePage(BaseModel):
    """Paginated response from searching a vector store."""

    object: str = "vector_store.search_results.page"
    search_query: str
    data: list[VectorStoreSearchResponse]
    has_more: bool = False
    next_page: str | None = None


@json_schema_type
class VectorStoreDeleteResponse(BaseModel):
    """Response from deleting a vector store."""

    id: str
    object: str = "vector_store.deleted"
    deleted: bool = True


@json_schema_type
class VectorStoreChunkingStrategyAuto(BaseModel):
    """Automatic chunking strategy for vector store files."""

    type: Literal["auto"] = "auto"


@json_schema_type
class VectorStoreChunkingStrategyStaticConfig(BaseModel):
    """Configuration for static chunking strategy."""

    chunk_overlap_tokens: int = 400
    max_chunk_size_tokens: int = Field(800, ge=100, le=4096)


@json_schema_type
class VectorStoreChunkingStrategyStatic(BaseModel):
    """Static chunking strategy with configurable parameters."""

    type: Literal["static"] = "static"
    static: VectorStoreChunkingStrategyStaticConfig


VectorStoreChunkingStrategy = Annotated[
    VectorStoreChunkingStrategyAuto | VectorStoreChunkingStrategyStatic,
    Field(discriminator="type"),
]
register_schema(VectorStoreChunkingStrategy, name="VectorStoreChunkingStrategy")


class SearchRankingOptions(BaseModel):
    """Options for ranking and filtering search results."""

    ranker: str | None = None
    # NOTE: OpenAI File Search Tool requires threshold to be between 0 and 1, however
    # we don't guarantee that the score is between 0 and 1, so will leave this unconstrained
    # and let the provider handle it
    score_threshold: float | None = Field(default=0.0)


@json_schema_type
class VectorStoreFileLastError(BaseModel):
    """Error information for failed vector store file processing."""

    code: Literal["server_error"] | Literal["rate_limit_exceeded"]
    message: str


VectorStoreFileStatus = Literal["completed"] | Literal["in_progress"] | Literal["cancelled"] | Literal["failed"]
register_schema(VectorStoreFileStatus, name="VectorStoreFileStatus")


@json_schema_type
class VectorStoreFileObject(BaseModel):
    """OpenAI Vector Store File object."""

    id: str
    object: str = "vector_store.file"
    attributes: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: VectorStoreChunkingStrategy
    created_at: int
    last_error: VectorStoreFileLastError | None = None
    status: VectorStoreFileStatus
    usage_bytes: int = 0
    vector_store_id: str


@json_schema_type
class VectorStoreListFilesResponse(BaseModel):
    """Response from listing files in a vector store."""

    object: str = "list"
    data: list[VectorStoreFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


@json_schema_type
class VectorStoreFileContentsResponse(BaseModel):
    """Response from retrieving the contents of a vector store file."""

    file_id: str
    filename: str
    attributes: dict[str, Any]
    content: list[VectorStoreContent]


@json_schema_type
class VectorStoreFileDeleteResponse(BaseModel):
    """Response from deleting a vector store file."""

    id: str
    object: str = "vector_store.file.deleted"
    deleted: bool = True


@json_schema_type
class VectorStoreFileBatchObject(BaseModel):
    """OpenAI Vector Store File Batch object."""

    id: str
    object: str = "vector_store.file_batch"
    created_at: int
    vector_store_id: str
    status: VectorStoreFileStatus
    file_counts: VectorStoreFileCounts


@json_schema_type
class VectorStoreFilesListInBatchResponse(BaseModel):
    """Response from listing files in a vector store file batch."""

    object: str = "list"
    data: list[VectorStoreFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICreateVectorStoreRequestWithExtraBody(BaseModel, extra="allow"):
    """Request to create a vector store with extra_body support."""

    name: str | None = None
    file_ids: list[str] | None = None
    expires_after: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICreateVectorStoreFileBatchRequestWithExtraBody(BaseModel, extra="allow"):
    """Request to create a vector store file batch with extra_body support."""

    file_ids: list[str]
    attributes: dict[str, Any] | None = None
    chunking_strategy: VectorStoreChunkingStrategy | None = None


@json_schema_type
class InsertChunksRequest(BaseModel):
    """Request to insert chunks into a vector database."""

    vector_store_id: str = Field(..., description="The identifier of the vector database to insert the chunks into.")
    chunks: list[Chunk] = Field(..., description="The chunks to insert.")
    ttl_seconds: int | None = Field(None, description="The time to live of the chunks.")


@json_schema_type
class QueryChunksRequest(BaseModel):
    """Request to query chunks from a vector database."""

    vector_store_id: str = Field(..., description="The identifier of the vector database to query.")
    query: InterleavedContent = Field(..., description="The query to search for.")
    params: dict[str, Any] | None = Field(None, description="The parameters of the query.")
