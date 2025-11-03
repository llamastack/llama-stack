# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Protocol, runtime_checkable

from fastapi import File, Form, Response, UploadFile

from llama_stack.apis.common.responses import Order
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import (
    ExpiresAfter,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFilePurpose,
)


@runtime_checkable
@trace_protocol
class FileService(Protocol):
    """Files

    This API is used to upload documents that can be used with other Llama Stack APIs.
    """

    # OpenAI Files API Endpoints
    async def openai_upload_file(
        self,
        file: Annotated[UploadFile, File()],
        purpose: Annotated[OpenAIFilePurpose, Form()],
        expires_after: Annotated[ExpiresAfter | None, Form()] = None,
    ) -> OpenAIFileObject:
        """Upload file."""
        ...

    async def openai_list_files(
        self,
        after: str | None = None,
        limit: int | None = 10000,
        order: Order | None = Order.desc,
        purpose: OpenAIFilePurpose | None = None,
    ) -> ListOpenAIFileResponse:
        """List files."""
        ...

    async def openai_retrieve_file(
        self,
        file_id: str,
    ) -> OpenAIFileObject:
        """Retrieve file."""
        ...

    async def openai_delete_file(
        self,
        file_id: str,
    ) -> OpenAIFileDeleteResponse:
        """Delete file."""
        ...

    async def openai_retrieve_file_content(
        self,
        file_id: str,
    ) -> Response:
        """Retrieve file content."""
        ...
