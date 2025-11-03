# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import Body

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import (
    Chunk,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentsResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)


class VectorStoreTable(Protocol):
    def get_vector_store(self, vector_store_id: str) -> VectorStore | None: ...


@runtime_checkable
@trace_protocol
class VectorIOService(Protocol):
    vector_store_table: VectorStoreTable | None = None

    # this will just block now until chunks are inserted, but it should
    # probably return a Job instance which can be polled for completion
    # TODO: rename vector_store_id to vector_store_id once Stainless is working
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
    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        """Creates a vector store.

        Generate an OpenAI-compatible vector store with the given parameters.
        :returns: A VectorStoreObject representing the created vector store.
        """
        ...

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

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store.

        :param vector_store_id: The ID of the vector store to retrieve.
        :returns: A VectorStoreObject representing the vector store.
        """
        ...

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

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store.

        :param vector_store_id: The ID of the vector store to delete.
        :returns: A VectorStoreDeleteResponse indicating the deletion status.
        """
        ...

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = 10,
        ranking_options: SearchRankingOptions | None = None,
        rewrite_query: bool | None = False,
        search_mode: str | None = "vector",
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

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileContentsResponse:
        """Retrieves the contents of a vector store file.

        :param vector_store_id: The ID of the vector store containing the file to retrieve.
        :param file_id: The ID of the file to retrieve.
        :returns: A list of InterleavedContent representing the file contents.
        """
        ...

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
