# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import (
    InsertChunksRequest,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksRequest,
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
    VectorStoreModifyRequest,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from .vector_io_service import VectorIOService


def get_vector_io_service(request: Request) -> VectorIOService:
    """Dependency to get the vector io service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.vector_io not in impls:
        raise ValueError("Vector IO API implementation not found")
    return impls[Api.vector_io]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Vector IO"],
    responses=standard_responses,
)


@router.post(
    "/vector-io/insert",
    response_model=None,
    status_code=204,
    summary="Insert chunks into a vector database.",
    description="Insert chunks into a vector database.",
)
async def insert_chunks(
    body: InsertChunksRequest = Body(...),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> None:
    """Insert chunks into a vector database."""
    await svc.insert_chunks(vector_store_id=body.vector_store_id, chunks=body.chunks, ttl_seconds=body.ttl_seconds)


@router.post(
    "/vector-io/query",
    response_model=QueryChunksResponse,
    summary="Query chunks from a vector database.",
    description="Query chunks from a vector database.",
)
async def query_chunks(
    body: QueryChunksRequest = Body(...),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> QueryChunksResponse:
    """Query chunks from a vector database."""
    return await svc.query_chunks(vector_store_id=body.vector_store_id, query=body.query, params=body.params)


# OpenAI Vector Stores API endpoints
@router.post(
    "/vector_stores",
    response_model=VectorStoreObject,
    summary="Creates a vector store.",
    description="Creates a vector store.",
)
async def openai_create_vector_store(
    body: OpenAICreateVectorStoreRequestWithExtraBody = Body(...),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreObject:
    """Creates a vector store."""
    return await svc.openai_create_vector_store(params=body)


@router.get(
    "/vector_stores",
    response_model=VectorStoreListResponse,
    summary="Returns a list of vector stores.",
    description="Returns a list of vector stores.",
)
async def openai_list_vector_stores(
    limit: int | None = Query(
        20,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
        ge=1,
        le=100,
    ),
    order: str | None = Query(
        "desc",
        description="Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.",
    ),
    after: str | None = Query(
        None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."
    ),
    before: str | None = Query(
        None,
        description="A cursor for use in pagination. `before` is an object ID that defines your place in the list.",
    ),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreListResponse:
    """Returns a list of vector stores."""
    return await svc.openai_list_vector_stores(limit=limit, order=order, after=after, before=before)


@router.get(
    "/vector_stores/{vector_store_id}",
    response_model=VectorStoreObject,
    summary="Retrieves a vector store.",
    description="Retrieves a vector store.",
)
async def openai_retrieve_vector_store(
    vector_store_id: Annotated[str, FastAPIPath(..., description="The ID of the vector store to retrieve.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreObject:
    """Retrieves a vector store."""
    return await svc.openai_retrieve_vector_store(vector_store_id=vector_store_id)


@router.post(
    "/vector_stores/{vector_store_id}",
    response_model=VectorStoreObject,
    summary="Updates a vector store.",
    description="Updates a vector store.",
)
async def openai_update_vector_store(
    vector_store_id: Annotated[str, FastAPIPath(..., description="The ID of the vector store to update.")],
    body: VectorStoreModifyRequest = Body(...),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreObject:
    """Updates a vector store."""
    return await svc.openai_update_vector_store(
        vector_store_id=vector_store_id,
        name=body.name,
        expires_after=body.expires_after,
        metadata=body.metadata,
    )


@router.delete(
    "/vector_stores/{vector_store_id}",
    response_model=VectorStoreDeleteResponse,
    summary="Delete a vector store.",
    description="Delete a vector store.",
)
async def openai_delete_vector_store(
    vector_store_id: Annotated[str, FastAPIPath(..., description="The ID of the vector store to delete.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreDeleteResponse:
    """Delete a vector store."""
    return await svc.openai_delete_vector_store(vector_store_id=vector_store_id)


@router.post(
    "/vector_stores/{vector_store_id}/search",
    response_model=VectorStoreSearchResponsePage,
    summary="Search for chunks in a vector store.",
    description="Search for chunks in a vector store.",
)
async def openai_search_vector_store(
    vector_store_id: Annotated[str, FastAPIPath(..., description="The ID of the vector store to search.")],
    query: str | list[str] = Body(..., description="The query string or array for performing the search."),
    filters: dict[str, Any] | None = Body(
        None, description="Filters based on file attributes to narrow the search results."
    ),
    max_num_results: int | None = Body(
        10, description="Maximum number of results to return (1 to 50 inclusive, default 10).", ge=1, le=50
    ),
    ranking_options: SearchRankingOptions | None = Body(
        None, description="Ranking options for fine-tuning the search results."
    ),
    rewrite_query: bool = Body(
        False, description="Whether to rewrite the natural language query for vector search (default false)."
    ),
    search_mode: str | None = Body(
        "vector", description="The search mode to use - 'keyword', 'vector', or 'hybrid' (default 'vector')."
    ),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreSearchResponsePage:
    """Search for chunks in a vector store."""
    return await svc.openai_search_vector_store(
        vector_store_id=vector_store_id,
        query=query,
        filters=filters,
        max_num_results=max_num_results,
        ranking_options=ranking_options,
        rewrite_query=rewrite_query,
        search_mode=search_mode,
    )


@router.post(
    "/vector_stores/{vector_store_id}/files",
    response_model=VectorStoreFileObject,
    summary="Attach a file to a vector store.",
    description="Attach a file to a vector store.",
)
async def openai_attach_file_to_vector_store(
    vector_store_id: Annotated[str, FastAPIPath(..., description="The ID of the vector store to attach the file to.")],
    file_id: str = Body(..., description="The ID of the file to attach to the vector store."),
    attributes: dict[str, Any] | None = Body(
        None, description="The key-value attributes stored with the file, which can be used for filtering."
    ),
    chunking_strategy: VectorStoreChunkingStrategy | None = Body(
        None, description="The chunking strategy to use for the file."
    ),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileObject:
    """Attach a file to a vector store."""
    return await svc.openai_attach_file_to_vector_store(
        vector_store_id=vector_store_id,
        file_id=file_id,
        attributes=attributes,
        chunking_strategy=chunking_strategy,
    )


@router.get(
    "/vector_stores/{vector_store_id}/files",
    response_model=VectorStoreListFilesResponse,
    summary="List files in a vector store.",
    description="List files in a vector store.",
)
async def openai_list_files_in_vector_store(
    vector_store_id: Annotated[str, FastAPIPath(..., description="The ID of the vector store to list files from.")],
    limit: int | None = Query(
        20,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
        ge=1,
        le=100,
    ),
    order: str | None = Query(
        "desc",
        description="Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.",
    ),
    after: str | None = Query(
        None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."
    ),
    before: str | None = Query(
        None,
        description="A cursor for use in pagination. `before` is an object ID that defines your place in the list.",
    ),
    filter: VectorStoreFileStatus | None = Query(
        None, description="Filter by file status to only return files with the specified status."
    ),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreListFilesResponse:
    """List files in a vector store."""
    return await svc.openai_list_files_in_vector_store(
        vector_store_id=vector_store_id, limit=limit, order=order, after=after, before=before, filter=filter
    )


@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    response_model=VectorStoreFileObject,
    summary="Retrieves a vector store file.",
    description="Retrieves a vector store file.",
)
async def openai_retrieve_vector_store_file(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file to retrieve.")
    ],
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to retrieve.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileObject:
    """Retrieves a vector store file."""
    return await svc.openai_retrieve_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)


@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}/content",
    response_model=VectorStoreFileContentsResponse,
    summary="Retrieves the contents of a vector store file.",
    description="Retrieves the contents of a vector store file.",
)
async def openai_retrieve_vector_store_file_contents(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file to retrieve.")
    ],
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to retrieve.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileContentsResponse:
    """Retrieves the contents of a vector store file."""
    return await svc.openai_retrieve_vector_store_file_contents(vector_store_id=vector_store_id, file_id=file_id)


@router.post(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    response_model=VectorStoreFileObject,
    summary="Updates a vector store file.",
    description="Updates a vector store file.",
)
async def openai_update_vector_store_file(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file to update.")
    ],
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to update.")],
    attributes: dict[str, Any] = Body(..., description="The updated key-value attributes to store with the file."),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileObject:
    """Updates a vector store file."""
    return await svc.openai_update_vector_store_file(
        vector_store_id=vector_store_id, file_id=file_id, attributes=attributes
    )


@router.delete(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    response_model=VectorStoreFileDeleteResponse,
    summary="Delete a vector store file.",
    description="Delete a vector store file.",
)
async def openai_delete_vector_store_file(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file to delete.")
    ],
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to delete.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileDeleteResponse:
    """Delete a vector store file."""
    return await svc.openai_delete_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)


@router.post(
    "/vector_stores/{vector_store_id}/file_batches",
    response_model=VectorStoreFileBatchObject,
    summary="Create a vector store file batch.",
    description="Create a vector store file batch.",
)
async def openai_create_vector_store_file_batch(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store to create the file batch for.")
    ],
    body: OpenAICreateVectorStoreFileBatchRequestWithExtraBody = Body(...),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileBatchObject:
    """Create a vector store file batch."""
    return await svc.openai_create_vector_store_file_batch(vector_store_id=vector_store_id, params=body)


@router.get(
    "/vector_stores/{vector_store_id}/file_batches/{batch_id}",
    response_model=VectorStoreFileBatchObject,
    summary="Retrieve a vector store file batch.",
    description="Retrieve a vector store file batch.",
)
async def openai_retrieve_vector_store_file_batch(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file batch.")
    ],
    batch_id: Annotated[str, FastAPIPath(..., description="The ID of the file batch to retrieve.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileBatchObject:
    """Retrieve a vector store file batch."""
    return await svc.openai_retrieve_vector_store_file_batch(batch_id=batch_id, vector_store_id=vector_store_id)


@router.get(
    "/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
    response_model=VectorStoreFilesListInBatchResponse,
    summary="Returns a list of vector store files in a batch.",
    description="Returns a list of vector store files in a batch.",
)
async def openai_list_files_in_vector_store_file_batch(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file batch.")
    ],
    batch_id: Annotated[str, FastAPIPath(..., description="The ID of the file batch to list files from.")],
    after: str | None = Query(
        None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."
    ),
    before: str | None = Query(
        None,
        description="A cursor for use in pagination. `before` is an object ID that defines your place in the list.",
    ),
    filter: str | None = Query(
        None, description="Filter by file status. One of in_progress, completed, failed, cancelled."
    ),
    limit: int | None = Query(
        20,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
        ge=1,
        le=100,
    ),
    order: str | None = Query(
        "desc",
        description="Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.",
    ),
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFilesListInBatchResponse:
    """Returns a list of vector store files in a batch."""
    return await svc.openai_list_files_in_vector_store_file_batch(
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
    summary="Cancels a vector store file batch.",
    description="Cancels a vector store file batch.",
)
async def openai_cancel_vector_store_file_batch(
    vector_store_id: Annotated[
        str, FastAPIPath(..., description="The ID of the vector store containing the file batch.")
    ],
    batch_id: Annotated[str, FastAPIPath(..., description="The ID of the file batch to cancel.")],
    svc: VectorIOService = Depends(get_vector_io_service),
) -> VectorStoreFileBatchObject:
    """Cancels a vector store file batch."""
    return await svc.openai_cancel_vector_store_file_batch(batch_id=batch_id, vector_store_id=vector_store_id)


# For backward compatibility with the router registry system
def create_vector_io_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Vector IO API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.vector_io, create_vector_io_router)
