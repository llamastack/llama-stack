# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import time
import uuid
from typing import Any

from fastapi import UploadFile

from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.vector_store import make_overlapped_chunks
from llama_stack_api.file_processors import ProcessFileResponse
from llama_stack_api.vector_io import (
    Chunk,
    ChunkMetadata,
    VectorStoreChunkingStrategy,
)

from .config import PyPDFFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")


class PyPDFFileProcessor:
    """PyPDF-based file processor for PDF documents."""

    def __init__(self, config: PyPDFFileProcessorConfig, files_api=None) -> None:
        self.config = config
        self.files_api = files_api

    async def process_file(
        self,
        file: UploadFile | None = None,
        file_id: str | None = None,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> ProcessFileResponse:
        """Process a PDF file and return chunks."""

        # Validate input
        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        # Get PDF content
        if file:
            # Read from uploaded file
            content = await file.read()
            filename = file.filename or "uploaded_file.pdf"
        elif file_id:
            # Get file from file storage using Files API
            if not self.files_api:
                raise ValueError("Files API not available - cannot process file_id")

            from llama_stack_api.files import RetrieveFileContentRequest, RetrieveFileRequest

            # Get file metadata
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            # Get file content
            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = content_response.body
        else:
            raise ValueError("Neither file nor file_id provided")

        # Note: File size validation is handled by Files API upload limits

        # Extract text from PDF
        text_content = self._extract_pdf_text(content)

        # Clean text if configured
        if self.config.clean_text:
            text_content = self._clean_text(text_content)

        # Extract metadata if configured
        pdf_metadata = {}
        if self.config.extract_metadata:
            pdf_metadata = self._extract_pdf_metadata(content)

        # Create document ID - prefer file_id for stability, fallback to filename
        document_id = file_id if file_id else (filename or str(uuid.uuid4()))

        # Prepare document metadata (include filename and file_id)
        document_metadata = {
            "filename": filename,
            **pdf_metadata,
        }
        if file_id:
            document_metadata["file_id"] = file_id

        # Create chunks
        chunks = self._create_chunks(text_content, document_id, chunking_strategy, document_metadata)

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create response metadata
        response_metadata = {
            "processor": "pypdf",
            "processing_time_ms": processing_time_ms,
            "page_count": pdf_metadata.get("page_count", 0),
            "extraction_method": "pypdf",
            "file_size_bytes": len(content),
            **pdf_metadata,
        }

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _extract_pdf_text(self, pdf_data: bytes) -> str:
        """Extract text from PDF using PyPDF."""
        try:
            from pypdf import PdfReader
        except ImportError as err:
            raise ImportError("PyPDF is required for PDF processing. Install with: pip install pypdf") from err

        pdf_bytes = io.BytesIO(pdf_data)
        reader = PdfReader(pdf_bytes)

        # Handle password-protected PDFs
        if reader.is_encrypted:
            if self.config.password:
                reader.decrypt(self.config.password)
            else:
                raise ValueError("PDF is encrypted but no password provided")

        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)
            except Exception as e:
                # Log warning but continue processing other pages
                log.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue

        return "\n".join(text_parts)

    def _extract_pdf_metadata(self, pdf_data: bytes) -> dict[str, Any]:
        """Extract metadata from PDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return {}

        pdf_bytes = io.BytesIO(pdf_data)
        reader = PdfReader(pdf_bytes)

        # Handle password-protected PDFs
        if reader.is_encrypted:
            if self.config.password:
                reader.decrypt(self.config.password)
            else:
                raise ValueError("PDF is encrypted but no password provided")

        metadata: dict[str, Any] = {"page_count": len(reader.pages)}

        # Extract document metadata
        if reader.metadata:
            pdf_metadata = reader.metadata
            if pdf_metadata.title:
                metadata["title"] = str(pdf_metadata.title)
            if pdf_metadata.author:
                metadata["author"] = str(pdf_metadata.author)
            if pdf_metadata.subject:
                metadata["subject"] = str(pdf_metadata.subject)
            if pdf_metadata.creator:
                metadata["creator"] = str(pdf_metadata.creator)
            if pdf_metadata.producer:
                metadata["producer"] = str(pdf_metadata.producer)
            if pdf_metadata.creation_date:
                metadata["creation_date"] = str(pdf_metadata.creation_date)
            if pdf_metadata.modification_date:
                metadata["modification_date"] = str(pdf_metadata.modification_date)

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Strip whitespace and normalize
            cleaned_line = " ".join(line.split())
            if cleaned_line:  # Only keep non-empty lines
                cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines)

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Create chunks from text content.

        Chunking semantics:
        - chunking_strategy is None → return single chunk (no chunking)
        - chunking_strategy.type == "auto" → use configured defaults (derived from vector-io)
        - chunking_strategy.type == "static" → use provided values
        """

        if not chunking_strategy:
            # No chunking - return entire text as a single chunk
            # Use tiktoken directly for token counting to avoid dependency on Llama tokenizer
            try:
                import tiktoken

                enc = tiktoken.get_encoding("cl100k_base")
                tokens = enc.encode(text)
                token_count = len(tokens)
            except Exception:
                # Fallback to word-based estimation if tiktoken fails
                token_count = len(text.split()) * 4 // 3  # Rough approximation

            chunk_id = f"{document_id}_chunk_0"

            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_tokenizer="DEFAULT_TIKTOKEN_TOKENIZER",
                content_token_count=token_count,
            )

            chunk = Chunk(
                content=text,  # Simple string content
                chunk_id=chunk_id,
                metadata={
                    "document_id": document_id,
                    **document_metadata,
                },
                chunk_metadata=chunk_metadata,
            )
            return [chunk]

        # Determine chunk parameters based on strategy
        if chunking_strategy.type == "auto":
            # Use configured defaults for auto chunking
            chunk_size = self.config.default_chunk_size_tokens
            overlap_size = self.config.default_chunk_overlap_tokens
        else:
            # Use provided static configuration
            chunk_size = chunking_strategy.static.max_chunk_size_tokens
            overlap_size = chunking_strategy.static.chunk_overlap_tokens

        # Prepare metadata for chunks (include filename and file_id)
        chunks_metadata_dict: dict[str, Any] = {
            "document_id": document_id,
            **document_metadata,
        }

        # Create overlapped chunks using existing utility (returns Chunk objects directly)
        chunks = make_overlapped_chunks(
            document_id=document_id,
            text=text,
            window_len=chunk_size,
            overlap_len=overlap_size,
            metadata=chunks_metadata_dict,
        )

        return chunks
