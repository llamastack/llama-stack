# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ProcessedContent(BaseModel):
    """
    Result of file processing containing extracted content and metadata.
    
    :param content: Extracted text content from the file
    :param metadata: Processing metadata including processor info, timing, etc.
    """
    content: str = Field(..., description="Extracted text content from file")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


@runtime_checkable
@trace_protocol
class FileProcessors(Protocol):
    """File Processors
    
    This API provides document processing capabilities for extracting text content
    from various file formats including PDFs, Word documents, and more.
    """

    @webmethod(route="/file-processors/process", method="POST", level=LLAMA_STACK_API_V1)
    async def process_file(
        self, 
        file_data: bytes, 
        filename: str,
        options: dict[str, Any] | None = None
    ) -> ProcessedContent:
        """Process a file and return extracted text content.
        
        :param file_data: The raw file data as bytes
        :param filename: Name of the file (used for format detection)
        :param options: Optional processing options (processor-specific)
        :returns: ProcessedContent with extracted text and metadata
        """