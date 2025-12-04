# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .schema_utils import json_schema_type, webmethod
from .vector_io import Chunk, VectorStoreChunkingStrategy
from .version import LLAMA_STACK_API_V1ALPHA


@json_schema_type
class ProcessFileRequest(BaseModel):
    """Request for processing a file into structured content.

    Exactly one of file_data or file_id must be provided.
    """

    file_data: bytes | str | None = None
    """Raw file data to process. Can be bytes for binary files or str for plain text content. Mutually exclusive with file_id."""

    file_id: str | None = None
    """ID of file already uploaded to file storage. Mutually exclusive with file_data."""

    filename: str
    """Original filename for format detection and processing hints."""

    options: dict[str, Any] | None = None
    """Optional processing options. Provider-specific parameters."""

    chunking_strategy: VectorStoreChunkingStrategy | None = None
    """Optional chunking strategy for splitting content into chunks."""

    generate_embeddings: bool = False
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
    """Processing-run metadata such as processor name/version, processing_time_ms,
    page_count, extraction_method (e.g. docling/pypdf/ocr), confidence scores,
    plus provider-specific fields."""


@runtime_checkable
class FileProcessors(Protocol):
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

    @webmethod(route="/file-processors/process", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def process_file(
        self,
        filename: str,
        file_data: bytes | str | None = None,
        file_id: str | None = None,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
        generate_embeddings: bool = False,
    ) -> ProcessedContent:
        """
        Process a file into structured content with optional chunking and embeddings.

        This method supports two modes of operation:
        1. Direct input: Process raw file data directly (file_data parameter)
        2. File storage: Process files already uploaded to file storage (file_id parameter)

        Exactly one of file_data or file_id must be provided.

        :param filename: Original filename for format detection and processing hints.
        :param file_data: Raw file data to process. Can be bytes for binary files or str for plain text content. Mutually exclusive with file_id.
        :param file_id: ID of file already uploaded to file storage. Mutually exclusive with file_data.
        :param options: Provider-specific processing options (e.g., OCR settings, output format).
        :param chunking_strategy: Optional strategy for splitting content into chunks.
        :param generate_embeddings: Whether to generate embeddings for chunks.
        :returns: ProcessedContent with extracted text, optional chunks, and metadata.
        """
        ...
