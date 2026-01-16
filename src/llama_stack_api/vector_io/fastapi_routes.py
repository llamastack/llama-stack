# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the VectorIO API.

This module defines the FastAPI router for the VectorIO API using standard
FastAPI route decorators.

It replaces the legacy @webmethod-driven route discovery for VectorIO.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, Query, Response, status
from pydantic import BaseModel

from llama_stack_api.common.content_types import InterleavedContent
from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1

from . import (
    EmbeddedChunk,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
    VectorStoreChunkingStrategy,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)


class _InsertChunksRequest(BaseModel):
    vector_store_id: str
    chunks: list[EmbeddedChunk]
    ttl_seconds: int | None = None


class _QueryChunksRequest(BaseModel):
    vector_store_id: str
    query: InterleavedContent
    params: dict[str, Any] | None = None


class _OpenAIUpdateVectorStoreRequest(BaseModel):
    name: str | None = None
    expires_after: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class _OpenAISearchVectorStoreRequest(BaseModel):
    query: str | list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int | None = 10
    ranking_options: SearchRankingOptions | None = None
    rewrite_query: bool | None = False
    search_mode: str | None = "vector"


class _OpenAIAttachFileRequest(BaseModel):
    file_id: str
    attributes: dict[str, Any] | None = None
    chunking_strategy: VectorStoreChunkingStrategy | None = None


class _OpenAIUpdateVectorStoreFileRequest(BaseModel):
    attributes: dict[str, Any]


def create_router(impl: VectorIO) -> APIRouter:
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["VectorIO"],
        responses=standard_responses,
    )

    @router.post(
        "/vector-io/insert",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        summary="Insert embedded chunks into a vector database.",
        description="Insert embedded chunks into a vector database.",
        responses={204: {"description": "Chunks were inserted."}},
    )
    async def insert_chunks(request: Annotated[_InsertChunksRequest, Body(...)]) -> None:
        await impl.insert_chunks(
            vector_store_id=request.vector_store_id,
            chunks=request.chunks,
            ttl_seconds=request.ttl_seconds,
        )
        return None

    @router.post(
        "/vector-io/query",
        response_model=QueryChunksResponse,
        summary="Query chunks from a vector database.",
        description="Query chunks from a vector database.",
        responses={200: {"description": "A QueryChunksResponse."}},
    )
    async def query_chunks(request: Annotated[_QueryChunksRequest, Body(...)]) -> QueryChunksResponse:
        return await impl.query_chunks(
            vector_store_id=request.vector_store_id,
            query=request.query,
            params=request.params,
        )

    @router.post(
        "/vector_stores",
        response_model=VectorStoreObject,
        summary="Create a vector store (OpenAI-compatible).",
        description="Create a vector store (OpenAI-compatible).",
        responses={200: {"description": "The created vector store."}},
    )
    async def openai_create_vector_store(
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        return await impl.openai_create_vector_store(params)

    @router.get(
        "/vector_stores",
        response_model=VectorStoreListResponse,
        summary="List vector stores (OpenAI-compatible).",
        description="List vector stores (OpenAI-compatible).",
        responses={200: {"description": "A list of vector stores."}},
    )
    async def openai_list_vector_stores(
        limit: Annotated[int | None, Query(description="Maximum number of vector stores to return.")] = 20,
        order: Annotated[str | None, Query(description="Sort order by created_at: asc or desc.")] = "desc",
        after: Annotated[
            str | None,
            Query(
                description="Pagination cursor (after).",
            ),
        ] = None,
        before: Annotated[
            str | None,
            Query(
                description="Pagination cursor (before).",
            ),
        ] = None,
    ) -> VectorStoreListResponse:
        return await impl.openai_list_vector_stores(limit=limit, order=order, after=after, before=before)

    @router.get(
        "/vector_stores/{vector_store_id}",
        response_model=VectorStoreObject,
        summary="Retrieve a vector store (OpenAI-compatible).",
        description="Retrieve a vector store (OpenAI-compatible).",
        responses={200: {"description": "The vector store."}},
    )
    async def openai_retrieve_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
    ) -> VectorStoreObject:
        return await impl.openai_retrieve_vector_store(vector_store_id=vector_store_id)

    @router.post(
        "/vector_stores/{vector_store_id}",
        response_model=VectorStoreObject,
        summary="Update a vector store (OpenAI-compatible).",
        description="Update a vector store (OpenAI-compatible).",
        responses={200: {"description": "The updated vector store."}},
    )
    async def openai_update_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        request: Annotated[_OpenAIUpdateVectorStoreRequest, Body(...)],
    ) -> VectorStoreObject:
        return await impl.openai_update_vector_store(
            vector_store_id=vector_store_id,
            name=request.name,
            expires_after=request.expires_after,
            metadata=request.metadata,
        )

    @router.delete(
        "/vector_stores/{vector_store_id}",
        response_model=VectorStoreDeleteResponse,
        summary="Delete a vector store (OpenAI-compatible).",
        description="Delete a vector store (OpenAI-compatible).",
        responses={200: {"description": "Vector store deleted."}},
    )
    async def openai_delete_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
    ) -> VectorStoreDeleteResponse:
        return await impl.openai_delete_vector_store(vector_store_id=vector_store_id)

    @router.post(
        "/vector_stores/{vector_store_id}/search",
        response_model=VectorStoreSearchResponsePage,
        summary="Search a vector store (OpenAI-compatible).",
        description="Search a vector store (OpenAI-compatible).",
        responses={200: {"description": "Search results."}},
    )
    async def openai_search_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        request: Annotated[_OpenAISearchVectorStoreRequest, Body(...)],
    ) -> VectorStoreSearchResponsePage:
        return await impl.openai_search_vector_store(
            vector_store_id=vector_store_id,
            query=request.query,
            filters=request.filters,
            max_num_results=request.max_num_results,
            ranking_options=request.ranking_options,
            rewrite_query=request.rewrite_query,
            search_mode=request.search_mode,
        )

    @router.post(
        "/vector_stores/{vector_store_id}/files",
        response_model=VectorStoreFileObject,
        summary="Attach a file to a vector store (OpenAI-compatible).",
        description="Attach a file to a vector store (OpenAI-compatible).",
        responses={200: {"description": "The attached file."}},
    )
    async def openai_attach_file_to_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        request: Annotated[_OpenAIAttachFileRequest, Body(...)],
    ) -> VectorStoreFileObject:
        return await impl.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            file_id=request.file_id,
            attributes=request.attributes,
            chunking_strategy=request.chunking_strategy,
        )

    @router.get(
        "/vector_stores/{vector_store_id}/files",
        response_model=VectorStoreListFilesResponse,
        summary="List files in a vector store (OpenAI-compatible).",
        description="List files in a vector store (OpenAI-compatible).",
        responses={200: {"description": "A list of files in the vector store."}},
    )
    async def openai_list_files_in_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        limit: Annotated[int | None, Query(description="Maximum number of files to return.")] = 20,
        order: Annotated[str | None, Query(description="Sort order by created_at: asc or desc.")] = "desc",
        after: Annotated[
            str | None,
            Query(
                description="Pagination cursor (after).",
            ),
        ] = None,
        before: Annotated[
            str | None,
            Query(
                description="Pagination cursor (before).",
            ),
        ] = None,
        filter: Annotated[VectorStoreFileStatus | None, Query(description="Filter by file status.")] = None,
    ) -> VectorStoreListFilesResponse:
        return await impl.openai_list_files_in_vector_store(
            vector_store_id=vector_store_id,
            limit=limit,
            order=order,
            after=after,
            before=before,
            filter=filter,
        )

    @router.get(
        "/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=VectorStoreFileObject,
        summary="Retrieve a vector store file (OpenAI-compatible).",
        description="Retrieve a vector store file (OpenAI-compatible).",
        responses={200: {"description": "The vector store file."}},
    )
    async def openai_retrieve_vector_store_file(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
    ) -> VectorStoreFileObject:
        return await impl.openai_retrieve_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)

    @router.get(
        "/vector_stores/{vector_store_id}/files/{file_id}/content",
        response_model=VectorStoreFileContentResponse,
        summary="Retrieve vector store file contents (OpenAI-compatible).",
        description="Retrieve vector store file contents (OpenAI-compatible).",
        responses={200: {"description": "The vector store file contents."}},
    )
    async def openai_retrieve_vector_store_file_contents(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
        include_embeddings: Annotated[bool | None, Query(description="Include embedding vectors.")] = False,
        include_metadata: Annotated[bool | None, Query(description="Include chunk metadata.")] = False,
    ) -> VectorStoreFileContentResponse:
        return await impl.openai_retrieve_vector_store_file_contents(
            vector_store_id=vector_store_id,
            file_id=file_id,
            include_embeddings=include_embeddings,
            include_metadata=include_metadata,
        )

    @router.post(
        "/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=VectorStoreFileObject,
        summary="Update a vector store file (OpenAI-compatible).",
        description="Update a vector store file (OpenAI-compatible).",
        responses={200: {"description": "The updated vector store file."}},
    )
    async def openai_update_vector_store_file(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
        request: Annotated[_OpenAIUpdateVectorStoreFileRequest, Body(...)],
    ) -> VectorStoreFileObject:
        return await impl.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=request.attributes,
        )

    @router.delete(
        "/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=VectorStoreFileDeleteResponse,
        summary="Delete a vector store file (OpenAI-compatible).",
        description="Delete a vector store file (OpenAI-compatible).",
        responses={200: {"description": "The vector store file was deleted."}},
    )
    async def openai_delete_vector_store_file(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
    ) -> VectorStoreFileDeleteResponse:
        return await impl.openai_delete_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)

    @router.post(
        "/vector_stores/{vector_store_id}/file_batches",
        response_model=VectorStoreFileBatchObject,
        summary="Create a vector store file batch (OpenAI-compatible).",
        description="Create a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "The created file batch."}},
    )
    async def openai_create_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        return await impl.openai_create_vector_store_file_batch(vector_store_id=vector_store_id, params=params)

    @router.get(
        "/vector_stores/{vector_store_id}/file_batches/{batch_id}",
        response_model=VectorStoreFileBatchObject,
        summary="Retrieve a vector store file batch (OpenAI-compatible).",
        description="Retrieve a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "The file batch."}},
    )
    async def openai_retrieve_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        batch_id: Annotated[str, Path(description="The file batch identifier.")],
    ) -> VectorStoreFileBatchObject:
        return await impl.openai_retrieve_vector_store_file_batch(batch_id=batch_id, vector_store_id=vector_store_id)

    @router.get(
        "/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
        response_model=VectorStoreFilesListInBatchResponse,
        summary="List files in a vector store file batch (OpenAI-compatible).",
        description="List files in a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "A list of files in the file batch."}},
    )
    async def openai_list_files_in_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        batch_id: Annotated[str, Path(description="The file batch identifier.")],
        after: Annotated[
            str | None,
            Query(
                description="Pagination cursor (after).",
            ),
        ] = None,
        before: Annotated[
            str | None,
            Query(
                description="Pagination cursor (before).",
            ),
        ] = None,
        filter: Annotated[str | None, Query(description="Filter by file status.")] = None,
        limit: Annotated[int | None, Query(description="Maximum number of files to return.")] = 20,
        order: Annotated[str | None, Query(description="Sort order by created_at: asc or desc.")] = "desc",
    ) -> VectorStoreFilesListInBatchResponse:
        return await impl.openai_list_files_in_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
            after=after,
            before=before,
            filter=filter,
            limit=limit,
            order=order,
        )

    @router.post(
        "/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
        response_model=VectorStoreFileBatchObject,
        summary="Cancel a vector store file batch (OpenAI-compatible).",
        description="Cancel a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "The cancelled file batch."}},
    )
    async def openai_cancel_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        batch_id: Annotated[str, Path(description="The file batch identifier.")],
    ) -> VectorStoreFileBatchObject:
        return await impl.openai_cancel_vector_store_file_batch(batch_id=batch_id, vector_store_id=vector_store_id)

    return router
