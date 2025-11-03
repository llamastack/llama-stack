# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Depends, File, Form, Query, Request, Response, UploadFile
from fastapi import Path as FastAPIPath

from llama_stack.apis.common.responses import Order
from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .files_service import FileService
from .models import (
    ExpiresAfter,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFilePurpose,
)


def get_file_service(request: Request) -> FileService:
    """Dependency to get the file service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.files not in impls:
        raise ValueError("Files API implementation not found")
    return impls[Api.files]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Files"],
    responses=standard_responses,
)


@router.post(
    "/files",
    response_model=OpenAIFileObject,
    summary="Upload file.",
    description="Upload a file that can be used across various endpoints.",
)
async def openai_upload_file(
    file: Annotated[UploadFile, File(..., description="The File object to be uploaded.")],
    purpose: Annotated[OpenAIFilePurpose, Form(..., description="The intended purpose of the uploaded file.")],
    expires_after: Annotated[
        ExpiresAfter | None, Form(description="Optional form values describing expiration for the file.")
    ] = None,
    svc: FileService = Depends(get_file_service),
) -> OpenAIFileObject:
    """Upload a file."""
    return await svc.openai_upload_file(file=file, purpose=purpose, expires_after=expires_after)


@router.get(
    "/files",
    response_model=ListOpenAIFileResponse,
    summary="List files.",
    description="Returns a list of files that belong to the user's organization.",
)
async def openai_list_files(
    after: str | None = Query(
        None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."
    ),
    limit: int | None = Query(
        10000,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 10,000, and the default is 10,000.",
    ),
    order: Order | None = Query(
        Order.desc,
        description="Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.",
    ),
    purpose: OpenAIFilePurpose | None = Query(None, description="Only return files with the given purpose."),
    svc: FileService = Depends(get_file_service),
) -> ListOpenAIFileResponse:
    """List files."""
    return await svc.openai_list_files(after=after, limit=limit, order=order, purpose=purpose)


@router.get(
    "/files/{file_id}",
    response_model=OpenAIFileObject,
    summary="Retrieve file.",
    description="Returns information about a specific file.",
)
async def openai_retrieve_file(
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to use for this request.")],
    svc: FileService = Depends(get_file_service),
) -> OpenAIFileObject:
    """Retrieve file information."""
    return await svc.openai_retrieve_file(file_id=file_id)


@router.delete(
    "/files/{file_id}",
    response_model=OpenAIFileDeleteResponse,
    summary="Delete file.",
    description="Delete a file.",
)
async def openai_delete_file(
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to use for this request.")],
    svc: FileService = Depends(get_file_service),
) -> OpenAIFileDeleteResponse:
    """Delete a file."""
    return await svc.openai_delete_file(file_id=file_id)


@router.get(
    "/files/{file_id}/content",
    response_class=Response,
    summary="Retrieve file content.",
    description="Returns the contents of the specified file.",
)
async def openai_retrieve_file_content(
    file_id: Annotated[str, FastAPIPath(..., description="The ID of the file to use for this request.")],
    svc: FileService = Depends(get_file_service),
) -> Response:
    """Retrieve file content."""
    return await svc.openai_retrieve_file_content(file_id=file_id)


# For backward compatibility with the router registry system
def create_files_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Files API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.files, create_files_router)
