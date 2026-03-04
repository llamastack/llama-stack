# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
import time
import uuid
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from fastapi import UploadFile

from llama_stack.log import get_logger
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id
from llama_stack_api.file_processors import ProcessFileResponse
from llama_stack_api.files import RetrieveFileContentRequest, RetrieveFileRequest
from llama_stack_api.vector_io import (
    Chunk,
    ChunkMetadata,
    VectorStoreChunkingStrategy,
)

from .config import DoclingFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")


class DoclingFileProcessor:
    """Docling-based file processor for PDF documents with structure-aware chunking."""

    def __init__(self, config: DoclingFileProcessorConfig, files_api=None) -> None:
        self.config = config
        self.files_api = files_api

    async def process_file(
        self,
        file: UploadFile | None = None,
        file_id: str | None = None,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> ProcessFileResponse:
        """Process a PDF file using docling and return chunks."""

        # Validate input
        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        # Get PDF content
        if file:
            content = await file.read()
            if len(content) > self.config.max_file_size_bytes:
                raise ValueError(
                    f"File size {len(content)} bytes exceeds maximum of {self.config.max_file_size_bytes} bytes"
                )
            filename = file.filename or f"{uuid.uuid4()}.pdf"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = content_response.body

        # Write to temp file since DocumentConverter expects a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()

            converter = DocumentConverter()
            result = converter.convert(tmp.name)

        doc = result.document
        page_count = doc.num_pages()

        # Check max_page_count
        if page_count > self.config.max_page_count:
            raise ValueError(
                f"Document has {page_count} pages, exceeding maximum of {self.config.max_page_count} pages"
            )

        document_id = str(uuid.uuid4())

        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_metadata: dict[str, Any] = {
            "processor": "docling",
            "processing_time_ms": processing_time_ms,
            "page_count": page_count,
            "extraction_method": "docling",
            "file_size_bytes": len(content),
        }

        # Create chunks
        chunks = self._create_chunks(doc, document_id, chunking_strategy, document_metadata)

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _create_chunks(
        self,
        doc: Any,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Create chunks from a docling Document.

        Chunking semantics:
        - chunking_strategy is None -> return all text as a single chunk
        - chunking_strategy.type == "auto" -> HybridChunker with configured defaults
        - chunking_strategy.type == "static" -> HybridChunker with provided max_tokens
        """
        if not chunking_strategy:
            # No chunking - collect all text as a single chunk
            text = doc.export_to_markdown()
            if not text or not text.strip():
                return []

            chunk_id = generate_chunk_id(document_id, text)
            return [
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata={
                        "document_id": document_id,
                        **document_metadata,
                    },
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=len(text.split()),
                    ),
                )
            ]

        # Determine max_tokens based on strategy
        if chunking_strategy.type == "auto":
            max_tokens = self.config.default_chunk_size_tokens
        elif chunking_strategy.type == "static":
            max_tokens = chunking_strategy.static.max_chunk_size_tokens
        else:
            max_tokens = self.config.default_chunk_size_tokens

        # max_tokens is set on the tokenizer, not on HybridChunker directly
        default_chunker = HybridChunker()
        tokenizer = HuggingFaceTokenizer(
            tokenizer=default_chunker.tokenizer.tokenizer,
            max_tokens=max_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer)
        doc_chunks = list(chunker.chunk(doc))

        if not doc_chunks:
            return []

        chunks: list[Chunk] = []
        for i, doc_chunk in enumerate(doc_chunks):
            text = doc_chunk.text
            if not text or not text.strip():
                continue

            headings = getattr(doc_chunk, "headings", None)
            chunk_window = f"{i}"

            chunk_id = generate_chunk_id(document_id, text, chunk_window)

            meta: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }
            if headings:
                meta["headings"] = headings

            chunks.append(
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata=meta,
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=len(text.split()),
                        chunk_window=chunk_window,
                    ),
                )
            )

        return chunks
