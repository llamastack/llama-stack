# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for File Processors API responses.

This module defines the response models for the File Processors API
using Pydantic with Field descriptions for OpenAPI schema generation.

Request models are not needed for this API since it uses multipart form data
with individual parameters rather than a JSON request body.
"""

from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api.schema_utils import json_schema_type
from llama_stack_api.vector_io import Chunk


@json_schema_type
class ProcessFileResponse(BaseModel):
    """Response model for file processing operation.

    Returns a list of chunks ready for storage in vector databases.
    Each chunk contains the content and metadata.
    """

    chunks: list[Chunk] = Field(..., description="Processed chunks from the file. Always returns at least one chunk.")

    metadata: dict[str, Any] = Field(
        ...,
        description="Processing-run metadata such as processor name/version, processing_time_ms, page_count, extraction_method (e.g. docling/pypdf/ocr), confidence scores, plus provider-specific fields.",
    )


__all__ = [
    "ProcessFileResponse",
]
