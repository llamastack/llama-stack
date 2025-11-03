# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class OpenAIFilePurpose(StrEnum):
    """
    Valid purpose values for OpenAI Files API.
    """

    ASSISTANTS = "assistants"
    BATCH = "batch"
    # TODO: Add other purposes as needed


@json_schema_type
class OpenAIFileObject(BaseModel):
    """OpenAI File object as defined in the OpenAI Files API."""

    object: Literal["file"] = Field(default="file", description="The object type, which is always 'file'.")
    id: str = Field(..., description="The file identifier, which can be referenced in the API endpoints.")
    bytes: int = Field(..., description="The size of the file, in bytes.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the file was created.")
    expires_at: int = Field(..., description="The Unix timestamp (in seconds) for when the file expires.")
    filename: str = Field(..., description="The name of the file.")
    purpose: OpenAIFilePurpose = Field(..., description="The intended purpose of the file.")


@json_schema_type
class ExpiresAfter(BaseModel):
    """Control expiration of uploaded files."""

    MIN: ClassVar[int] = 3600  # 1 hour
    MAX: ClassVar[int] = 2592000  # 30 days

    anchor: Literal["created_at"] = Field(..., description="Anchor must be 'created_at'.")
    seconds: int = Field(..., ge=3600, le=2592000, description="Seconds between 3600 and 2592000 (1 hour to 30 days).")


@json_schema_type
class ListOpenAIFileResponse(BaseModel):
    """Response for listing files in OpenAI Files API."""

    data: list[OpenAIFileObject] = Field(..., description="List of file objects.")
    has_more: bool = Field(..., description="Whether there are more files available beyond this page.")
    first_id: str = Field(..., description="ID of the first file in the list for pagination.")
    last_id: str = Field(..., description="ID of the last file in the list for pagination.")
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")


@json_schema_type
class OpenAIFileDeleteResponse(BaseModel):
    """Response for deleting a file in OpenAI Files API."""

    id: str = Field(..., description="The file identifier that was deleted.")
    object: Literal["file"] = Field(default="file", description="The object type, which is always 'file'.")
    deleted: bool = Field(..., description="Whether the file was successfully deleted.")
