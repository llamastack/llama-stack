# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from fastapi import UploadFile

from llama_stack_api.vector_io import VectorStoreChunkingStrategy

from .models import ProcessFileResponse


@runtime_checkable
class FileProcessors(Protocol):
    """
    File Processor API for converting files into structured, processable content.

    This API provides a flexible interface for processing various file formats
    (PDFs, documents, images, etc.) into normalized text content that can be used for
    vector store ingestion, RAG applications, or standalone content extraction.

    The API focuses on parsing and normalization:
    - Multiple file formats through extensible provider architecture
    - Multipart form uploads or file ID references
    - Configurable processing options per provider
    - Optional chunking using provider's native capabilities
    - Rich metadata about processing results

    For embedding generation, use the chunks from this API with the separate
    embedding API to maintain clean separation of concerns.

    Future providers can extend this interface to support additional formats,
    processing capabilities, and optimization strategies.
    """

    async def process_file(
        self,
        file: UploadFile | None = None,
        file_id: str | None = None,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> ProcessFileResponse:
        """
        Process a file into chunks ready for vector database storage.

        This method supports two modes of operation via multipart form request:
        1. Direct upload: Upload and process a file directly (file parameter)
        2. File storage: Process files already uploaded to file storage (file_id parameter)

        Exactly one of file or file_id must be provided.

        If no chunking_strategy is provided, the entire file content is returned as a single chunk.
        If chunking_strategy is provided, the file is split according to the strategy.

        :param file: The uploaded file object containing content and metadata (filename, content_type, etc.). Mutually exclusive with file_id.
        :param file_id: ID of file already uploaded to file storage. Mutually exclusive with file.
        :param options: Provider-specific processing options (e.g., OCR settings, output format).
        :param chunking_strategy: Optional strategy for splitting content into chunks.
        :returns: ProcessFileResponse with chunks ready for vector database storage.
        """
        ...
