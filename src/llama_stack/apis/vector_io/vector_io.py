# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from fastapi import Body, Query
from pydantic import BaseModel, Field

from llama_stack.apis.common.tracing import telemetry_traceable
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, webmethod
from llama_stack.strong_typing.schema import register_schema


@json_schema_type
class ChunkMetadata(BaseModel):
    """
    `ChunkMetadata` is backend metadata for a `Chunk` that is used to store additional information about the chunk that
        will not be used in the context during inference, but is required for backend functionality. The `ChunkMetadata`
        is set during chunk creation in `MemoryToolRuntimeImpl().insert()`and is not expected to change after.
        Use `Chunk.metadata` for metadata that will be used in the context during inference.
    :param chunk_id: The ID of the chunk. If not set, it will be generated based on the document ID and content.
    :param document_id: The ID of the document this chunk belongs to.
    :param source: The source of the content, such as a URL, file path, or other identifier.
    :param created_timestamp: An optional timestamp indicating when the chunk was created.
    :param updated_timestamp: An optional timestamp indicating when the chunk was last updated.
    :param chunk_window: The window of the chunk, which can be used to group related chunks together.
    :param chunk_tokenizer: The tokenizer used to create the chunk. Default is Tiktoken.
    :param chunk_embedding_model: The embedding model used to create the chunk's embedding.
    :param chunk_embedding_dimension: The dimension of the embedding vector for the chunk.
    :param content_token_count: The number of tokens in the content of the chunk.
    :param metadata_token_count: The number of tokens in the metadata of the chunk.
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
    """
    A chunk of content that can be inserted into a vector database.
    :param content: The content of the chunk, which can be interleaved text, images, or other types.
    :param chunk_id: Unique identifier for the chunk. Must be provided explicitly.
    :param metadata: Metadata associated with the chunk that will be used in the model context during inference.
    :param embedding: Optional embedding for the chunk. If not provided, it will be computed later.
    :param chunk_metadata: Metadata for the chunk that will NOT be used in the context during inference.
        The `chunk_metadata` is required backend functionality.
    """

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
    """Response from querying chunks in a vector database.

    :param chunks: List of content chunks returned from the query
    :param scores: Relevance scores corresponding to each returned chunk
    """

    chunks: list[Chunk]
    scores: list[float]


@json_schema_type
class VectorStoreFileCounts(BaseModel):
    """File processing status counts for a vector store.

    :param completed: Number of files that have been successfully processed
    :param cancelled: Number of files that had their processing cancelled
    :param failed: Number of files that failed to process
    :param in_progress: Number of files currently being processed
    :param total: Total number of files in the vector store
    """

    completed: int
    cancelled: int
    failed: int
    in_progress: int
    total: int


# TODO: rename this as OpenAIVectorStore
@json_schema_type
class VectorStoreObject(BaseModel):
    """OpenAI Vector Store object.

    :param id: Unique identifier for the vector store
    :param object: Object type identifier, always "vector_store"
    :param created_at: Timestamp when the vector store was created
    :param name: (Optional) Name of the vector store
    :param usage_bytes: Storage space used by the vector store in bytes
    :param file_counts: File processing status counts for the vector store
    :param status: Current status of the vector store
    :param expires_after: (Optional) Expiration policy for the vector store
    :param expires_at: (Optional) Timestamp when the vector store will expire
    :param last_active_at: (Optional) Timestamp of last activity on the vector store
    :param metadata: Set of key-value pairs that can be attached to the vector store
    """

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
    """Request to create a vector store.

    :param name: (Optional) Name for the vector store
    :param file_ids: List of file IDs to include in the vector store
    :param expires_after: (Optional) Expiration policy for the vector store
    :param chunking_strategy: (Optional) Strategy for splitting files into chunks
    :param metadata: Set of key-value pairs that can be attached to the vector store
    """

    name: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    expires_after: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class VectorStoreModifyRequest(BaseModel):
    """Request to modify a vector store.

    :param name: (Optional) Updated name for the vector store
    :param expires_after: (Optional) Updated expiration policy for the vector store
    :param metadata: (Optional) Updated set of key-value pairs for the vector store
    """

    name: str | None = None
    expires_after: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreListResponse(BaseModel):
    """Response from listing vector stores.

    :param object: Object type identifier, always "list"
    :param data: List of vector store objects
    :param first_id: (Optional) ID of the first vector store in the list for pagination
    :param last_id: (Optional) ID of the last vector store in the list for pagination
    :param has_more: Whether there are more vector stores available beyond this page
    """

    object: str = "list"
    data: list[VectorStoreObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


@json_schema_type
class VectorStoreSearchRequest(BaseModel):
    """Request to search a vector store.

    :param query: Search query as a string or list of strings
    :param filters: (Optional) Filters based on file attributes to narrow search results
    :param max_num_results: Maximum number of results to return, defaults to 10
    :param ranking_options: (Optional) Options for ranking and filtering search results
    :param rewrite_query: Whether to rewrite the query for better vector search performance
    """

    query: str | list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int = 10
    ranking_options: dict[str, Any] | None = None
    rewrite_query: bool = False


@json_schema_type
class VectorStoreContent(BaseModel):
    """Content item from a vector store file or search result.

    :param type: Content type, currently only "text" is supported
    :param text: The actual text content
    :param embedding: Optional embedding vector for this content chunk
    :param chunk_metadata: Optional chunk metadata
    :param metadata: Optional user-defined metadata
    """

    type: Literal["text"]
    text: str
    embedding: list[float] | None = None
    chunk_metadata: ChunkMetadata | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreSearchResponse(BaseModel):
    """Response from searching a vector store.

    :param file_id: Unique identifier of the file containing the result
    :param filename: Name of the file containing the result
    :param score: Relevance score for this search result
    :param attributes: (Optional) Key-value attributes associated with the file
    :param content: List of content items matching the search query
    """

    file_id: str
    filename: str
    score: float
    attributes: dict[str, str | float | bool] | None = None
    content: list[VectorStoreContent]


@json_schema_type
class VectorStoreSearchResponsePage(BaseModel):
    """Paginated response from searching a vector store.

    :param object: Object type identifier for the search results page
    :param search_query: The original search query that was executed
    :param data: List of search result objects
    :param has_more: Whether there are more results available beyond this page
    :param next_page: (Optional) Token for retrieving the next page of results
    """

    object: str = "vector_store.search_results.page"
    search_query: list[str]
    data: list[VectorStoreSearchResponse]
    has_more: bool = False
    next_page: str | None = None


@json_schema_type
class VectorStoreDeleteResponse(BaseModel):
    """Response from deleting a vector store.

    :param id: Unique identifier of the deleted vector store
    :param object: Object type identifier for the deletion response
    :param deleted: Whether the deletion operation was successful
    """

    id: str
    object: str = "vector_store.deleted"
    deleted: bool = True


@json_schema_type
class VectorStoreFileContentResponse(BaseModel):
    """Represents the parsed content of a vector store file.

    :param object: The object type, which is always `vector_store.file_content.page`
    :param data: Parsed content of the file
    :param has_more: Indicates if there are more content pages to fetch
    :param next_page: The token for the next page, if any
    """

    object: Literal["vector_store.file_content.page"] = "vector_store.file_content.page"
    data: list[VectorStoreContent]
    has_more: bool = False
    next_page: str | None = None


@json_schema_type
class VectorStoreChunkingStrategyAuto(BaseModel):
    """Automatic chunking strategy for vector store files.

    :param type: Strategy type, always "auto" for automatic chunking
    """

    type: Literal["auto"] = "auto"


@json_schema_type
class VectorStoreChunkingStrategyStaticConfig(BaseModel):
    """Configuration for static chunking strategy.

    :param chunk_overlap_tokens: Number of tokens to overlap between adjacent chunks
    :param max_chunk_size_tokens: Maximum number of tokens per chunk, must be between 100 and 4096
    """

    chunk_overlap_tokens: int = 400
    max_chunk_size_tokens: int = Field(800, ge=100, le=4096)


@json_schema_type
class VectorStoreChunkingStrategyStatic(BaseModel):
    """Static chunking strategy with configurable parameters.

    :param type: Strategy type, always "static" for static chunking
    :param static: Configuration parameters for the static chunking strategy
    """

    type: Literal["static"] = "static"
    static: VectorStoreChunkingStrategyStaticConfig


VectorStoreChunkingStrategy = Annotated[
    VectorStoreChunkingStrategyAuto | VectorStoreChunkingStrategyStatic,
    Field(discriminator="type"),
]
register_schema(VectorStoreChunkingStrategy, name="VectorStoreChunkingStrategy")


class SearchRankingOptions(BaseModel):
    """Options for ranking and filtering search results.

    :param ranker: (Optional) Name of the ranking algorithm to use
    :param score_threshold: (Optional) Minimum relevance score threshold for results
    """

    ranker: str | None = None
    # NOTE: OpenAI File Search Tool requires threshold to be between 0 and 1, however
    # we don't guarantee that the score is between 0 and 1, so will leave this unconstrained
    # and let the provider handle it
    score_threshold: float | None = Field(default=0.0)


@json_schema_type
class VectorStoreFileLastError(BaseModel):
    """Error information for failed vector store file processing.

    :param code: Error code indicating the type of failure
    :param message: Human-readable error message describing the failure
    """

    code: Literal["server_error"] | Literal["rate_limit_exceeded"]
    message: str


VectorStoreFileStatus = Literal["completed"] | Literal["in_progress"] | Literal["cancelled"] | Literal["failed"]
register_schema(VectorStoreFileStatus, name="VectorStoreFileStatus")


@json_schema_type
class VectorStoreFileObject(BaseModel):
    """OpenAI Vector Store File object.

    :param id: Unique identifier for the file
    :param object: Object type identifier, always "vector_store.file"
    :param attributes: Key-value attributes associated with the file
    :param chunking_strategy: Strategy used for splitting the file into chunks
    :param created_at: Timestamp when the file was added to the vector store
    :param last_error: (Optional) Error information if file processing failed
    :param status: Current processing status of the file
    :param usage_bytes: Storage space used by this file in bytes
    :param vector_store_id: ID of the vector store containing this file
    """

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
    """Response from listing files in a vector store.

    :param object: Object type identifier, always "list"
    :param data: List of vector store file objects
    :param first_id: (Optional) ID of the first file in the list for pagination
    :param last_id: (Optional) ID of the last file in the list for pagination
    :param has_more: Whether there are more files available beyond this page
    """

    object: str = "list"
    data: list[VectorStoreFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


@json_schema_type
class VectorStoreFileDeleteResponse(BaseModel):
    """Response from deleting a vector store file.

    :param id: Unique identifier of the deleted file
    :param object: Object type identifier for the deletion response
    :param deleted: Whether the deletion operation was successful
    """

    id: str
    object: str = "vector_store.file.deleted"
    deleted: bool = True


@json_schema_type
class VectorStoreFileBatchObject(BaseModel):
    """OpenAI Vector Store File Batch object.

    :param id: Unique identifier for the file batch
    :param object: Object type identifier, always "vector_store.file_batch"
    :param created_at: Timestamp when the file batch was created
    :param vector_store_id: ID of the vector store containing the file batch
    :param status: Current processing status of the file batch
    :param file_counts: File processing status counts for the batch
    """

    id: str
    object: str = "vector_store.file_batch"
    created_at: int
    vector_store_id: str
    status: VectorStoreFileStatus
    file_counts: VectorStoreFileCounts


@json_schema_type
class VectorStoreFilesListInBatchResponse(BaseModel):
    """Response from listing files in a vector store file batch.

    :param object: Object type identifier, always "list"
    :param data: List of vector store file objects in the batch
    :param first_id: (Optional) ID of the first file in the list for pagination
    :param last_id: (Optional) ID of the last file in the list for pagination
    :param has_more: Whether there are more files available beyond this page
    """

    object: str = "list"
    data: list[VectorStoreFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICreateVectorStoreRequestWithExtraBody(BaseModel, extra="allow"):
    """Request to create a vector store with extra_body support.

    :param name: (Optional) A name for the vector store
    :param file_ids: List of file IDs to include in the vector store
    :param expires_after: (Optional) Expiration policy for the vector store
    :param chunking_strategy: (Optional) Strategy for splitting files into chunks
    :param metadata: Set of key-value pairs that can be attached to the vector store
    """

    name: str | None = None
    file_ids: list[str] | None = None
    expires_after: dict[str, Any] | None = None
    chunking_strategy: VectorStoreChunkingStrategy | None = None
    metadata: dict[str, Any] | None = None


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICreateVectorStoreFileBatchRequestWithExtraBody(BaseModel, extra="allow"):
    """Request to create a vector store file batch with extra_body support.

    :param file_ids: A list of File IDs that the vector store should use
    :param attributes: (Optional) Key-value attributes to store with the files
    :param chunking_strategy: (Optional) The chunking strategy used to chunk the file(s). Defaults to auto
    """

    file_ids: list[str]
    attributes: dict[str, Any] | None = None
    chunking_strategy: VectorStoreChunkingStrategy | None = None


class VectorStoreTable(Protocol):
    def get_vector_store(self, vector_store_id: str) -> VectorStore | None: ...


@runtime_checkable
@telemetry_traceable
class VectorIO(Protocol):
    vector_store_table: VectorStoreTable | None = None

    # this will just block now until chunks are inserted, but it should
    # probably return a Job instance which can be polled for completion
    # TODO: rename vector_store_id to vector_store_id once Stainless is working
    @webmethod(route="/vector-io/insert", method="POST", level=LLAMA_STACK_API_V1)
    async def insert_chunks(
        self,
        vector_store_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert chunks into a vector database.

        :param vector_store_id: The identifier of the vector database to insert the chunks into.
        :param chunks: The chunks to insert. Each `Chunk` should contain content which can be interleaved text, images, or other types.
            `metadata`: `dict[str, Any]` and `embedding`: `List[float]` are optional.
            If `metadata` is provided, you configure how Llama Stack formats the chunk during generation.
            If `embedding` is not provided, it will be computed later.
        :param ttl_seconds: The time to live of the chunks.
        """
        ...

    # TODO: rename vector_store_id to vector_store_id once Stainless is working
    @webmethod(route="/vector-io/query", method="POST", level=LLAMA_STACK_API_V1)
    async def query_chunks(
        self,
        vector_store_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Query chunks from a vector database.

        :param vector_store_id: The identifier of the vector database to query.
        :param query: The query to search for.
        :param params: The parameters of the query.
        :returns: A QueryChunksResponse.
        """
        ...

    # OpenAI Vector Stores API endpoints
    @webmethod(route="/vector_stores", method="POST", level=LLAMA_STACK_API_V1)
    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        """Creates a vector store.

        Generate an OpenAI-compatible vector store with the given parameters.
        :returns: A VectorStoreObject representing the created vector store.
        """
        ...

    @webmethod(route="/vector_stores", method="GET", level=LLAMA_STACK_API_V1)
    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        """Returns a list of vector stores.

        :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
        :param order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.
        :param after: A cursor for use in pagination. `after` is an object ID that defines your place in the list.
        :param before: A cursor for use in pagination. `before` is an object ID that defines your place in the list.
        :returns: A VectorStoreListResponse containing the list of vector stores.
        """
        ...

    @webmethod(route="/vector_stores/{vector_store_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store.

        :param vector_store_id: The ID of the vector store to retrieve.
        :returns: A VectorStoreObject representing the vector store.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}",
        method="POST",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        """Updates a vector store.

        :param vector_store_id: The ID of the vector store to update.
        :param name: The name of the vector store.
        :param expires_after: The expiration policy for a vector store.
        :param metadata: Set of 16 key-value pairs that can be attached to an object.
        :returns: A VectorStoreObject representing the updated vector store.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}",
        method="DELETE",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store.

        :param vector_store_id: The ID of the vector store to delete.
        :returns: A VectorStoreDeleteResponse indicating the deletion status.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/search",
        method="POST",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = 10,
        ranking_options: SearchRankingOptions | None = None,
        rewrite_query: bool | None = False,
        search_mode: (
            str | None
        ) = "vector",  # Using str instead of Literal due to OpenAPI schema generator limitations
    ) -> VectorStoreSearchResponsePage:
        """Search for chunks in a vector store.

        Searches a vector store for relevant chunks based on a query and optional file attribute filters.

        :param vector_store_id: The ID of the vector store to search.
        :param query: The query string or array for performing the search.
        :param filters: Filters based on file attributes to narrow the search results.
        :param max_num_results: Maximum number of results to return (1 to 50 inclusive, default 10).
        :param ranking_options: Ranking options for fine-tuning the search results.
        :param rewrite_query: Whether to rewrite the natural language query for vector search (default false)
        :param search_mode: The search mode to use - "keyword", "vector", or "hybrid" (default "vector")
        :returns: A VectorStoreSearchResponse containing the search results.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/files",
        method="POST",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> VectorStoreFileObject:
        """Attach a file to a vector store.

        :param vector_store_id: The ID of the vector store to attach the file to.
        :param file_id: The ID of the file to attach to the vector store.
        :param attributes: The key-value attributes stored with the file, which can be used for filtering.
        :param chunking_strategy: The chunking strategy to use for the file.
        :returns: A VectorStoreFileObject representing the attached file.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/files",
        method="GET",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        """List files in a vector store.

        :param vector_store_id: The ID of the vector store to list files from.
        :param limit: (Optional) A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
        :param order: (Optional) Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.
        :param after: (Optional) A cursor for use in pagination. `after` is an object ID that defines your place in the list.
        :param before: (Optional) A cursor for use in pagination. `before` is an object ID that defines your place in the list.
        :param filter: (Optional) Filter by file status to only return files with the specified status.
        :returns: A VectorStoreListFilesResponse containing the list of files.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/files/{file_id}",
        method="GET",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        """Retrieves a vector store file.

        :param vector_store_id: The ID of the vector store containing the file to retrieve.
        :param file_id: The ID of the file to retrieve.
        :returns: A VectorStoreFileObject representing the file.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/files/{file_id}/content",
        method="GET",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
        include_embeddings: Annotated[bool | None, Query(default=False)] = False,
        include_metadata: Annotated[bool | None, Query(default=False)] = False,
    ) -> VectorStoreFileContentResponse:
        """Retrieves the contents of a vector store file.

        :param vector_store_id: The ID of the vector store containing the file to retrieve.
        :param file_id: The ID of the file to retrieve.
        :param include_embeddings: Whether to include embedding vectors in the response.
        :param include_metadata: Whether to include chunk metadata in the response.
        :returns: File contents, optionally with embeddings and metadata based on query parameters.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/files/{file_id}",
        method="POST",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any],
    ) -> VectorStoreFileObject:
        """Updates a vector store file.

        :param vector_store_id: The ID of the vector store containing the file to update.
        :param file_id: The ID of the file to update.
        :param attributes: The updated key-value attributes to store with the file.
        :returns: A VectorStoreFileObject representing the updated file.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/files/{file_id}",
        method="DELETE",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        """Delete a vector store file.

        :param vector_store_id: The ID of the vector store containing the file to delete.
        :param file_id: The ID of the file to delete.
        :returns: A VectorStoreFileDeleteResponse indicating the deletion status.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/file_batches",
        method="POST",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        """Create a vector store file batch.

        Generate an OpenAI-compatible vector store file batch for the given vector store.
        :param vector_store_id: The ID of the vector store to create the file batch for.
        :returns: A VectorStoreFileBatchObject representing the created file batch.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/file_batches/{batch_id}",
        method="GET",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Retrieve a vector store file batch.

        :param batch_id: The ID of the file batch to retrieve.
        :param vector_store_id: The ID of the vector store containing the file batch.
        :returns: A VectorStoreFileBatchObject representing the file batch.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
        method="GET",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_list_files_in_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
        after: str | None = None,
        before: str | None = None,
        filter: str | None = None,
        limit: int | None = 20,
        order: str | None = "desc",
    ) -> VectorStoreFilesListInBatchResponse:
        """Returns a list of vector store files in a batch.

        :param batch_id: The ID of the file batch to list files from.
        :param vector_store_id: The ID of the vector store containing the file batch.
        :param after: A cursor for use in pagination. `after` is an object ID that defines your place in the list.
        :param before: A cursor for use in pagination. `before` is an object ID that defines your place in the list.
        :param filter: Filter by file status. One of in_progress, completed, failed, cancelled.
        :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
        :param order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.
        :returns: A VectorStoreFilesListInBatchResponse containing the list of files in the batch.
        """
        ...

    @webmethod(
        route="/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
        method="POST",
        level=LLAMA_STACK_API_V1,
    )
    async def openai_cancel_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Cancels a vector store file batch.

        :param batch_id: The ID of the file batch to cancel.
        :param vector_store_id: The ID of the vector store containing the file batch.
        :returns: A VectorStoreFileBatchObject representing the cancelled file batch.
        """
        ...
