# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .common.tracing import telemetry_traceable
from .schema_utils import json_schema_type, webmethod
from .vector_io import Chunk, VectorStoreChunkingStrategy
from .version import LLAMA_STACK_API_V1ALPHA


@json_schema_type
class ProcessFileRequest(BaseModel):
    """Request for processing a file into structured content."""

    file_data: bytes
    """Raw file data to process."""

    filename: str
    """Original filename for format detection and processing hints."""

    options: dict[str, Any] | None = None
    """Optional processing options. Provider-specific parameters."""

    chunking_strategy: VectorStoreChunkingStrategy | None = None
    """Optional chunking strategy for splitting content into chunks."""

    include_embeddings: bool = False
    """Whether to generate embeddings for chunks."""


@json_schema_type
class ProcessedContent(BaseModel):
    """Result of file processing operation."""

    content: str
    """Extracted text content from the file."""

    chunks: list[Chunk] | None = None
    """Optional chunks if chunking strategy was provided."""

    embeddings: list[list[float]] | None = None
    """Optional embeddings for chunks if requested."""

    metadata: dict[str, Any]
    """Processing metadata including processor name, timing, and provider-specific data."""


@telemetry_traceable
@runtime_checkable
class FileProcessor(Protocol):
    """
    File Processor API for converting files into structured, processable content.

    This API provides a flexible interface for processing various file formats
    (PDFs, documents, images, etc.) into text content that can be used for
    vector store ingestion, RAG applications, or standalone content extraction.

    The API supports:
    - Multiple file formats through extensible provider architecture
    - Configurable processing options per provider
    - Integration with vector store chunking strategies
    - Optional embedding generation for chunks
    - Rich metadata about processing results

    Future providers can extend this interface to support additional formats,
    processing capabilities, and optimization strategies.
    """

    @webmethod(route="/file-processor/process", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def process_file(
        self,
        file_data: bytes,
        filename: str,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
        include_embeddings: bool = False,
    ) -> ProcessedContent:
        """
        Process a file into structured content with optional chunking and embeddings.

        This method processes raw file data and converts it into text content for applications such as vector store ingestion.

        :param file_data: Raw bytes of the file to process.
        :param filename: Original filename for format detection.
        :param options: Provider-specific processing options (e.g., OCR settings, output format).
        :param chunking_strategy: Optional strategy for splitting content into chunks.
        :param include_embeddings: Whether to generate embeddings for chunks.
        :returns: ProcessedContent with extracted text, optional chunks, and metadata.
        """
        ...
