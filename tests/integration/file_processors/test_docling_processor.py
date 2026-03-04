# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import uuid
from pathlib import Path

import pytest
from fastapi import UploadFile

pytest.importorskip("docling", reason="docling is not installed")

from llama_stack.providers.inline.file_processor.docling import DoclingFileProcessorConfig  # noqa: E402
from llama_stack.providers.inline.file_processor.docling.docling import DoclingFileProcessor  # noqa: E402
from llama_stack_api.vector_io import (
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)


class TestDoclingFileProcessor:
    """Integration tests for Docling file processor."""

    @pytest.fixture
    def config(self) -> DoclingFileProcessorConfig:
        """Default configuration for testing."""
        return DoclingFileProcessorConfig(
            default_chunk_size_tokens=512,
            default_chunk_overlap_tokens=50,
        )

    @pytest.fixture
    def processor(self, config: DoclingFileProcessorConfig) -> DoclingFileProcessor:
        """Docling processor instance for testing."""
        return DoclingFileProcessor(config, files_api=None)

    @pytest.fixture
    def test_pdf_path(self) -> Path:
        """Path to the test PDF file."""
        return Path(__file__).resolve().parents[1] / "responses" / "fixtures" / "pdfs" / "llama_stack_and_models.pdf"

    @pytest.fixture
    def test_pdf_content(self, test_pdf_path: Path) -> bytes:
        """Content of the test PDF file."""
        with open(test_pdf_path, "rb") as f:
            return f.read()

    @pytest.fixture
    def upload_file(self, test_pdf_content: bytes) -> UploadFile:
        """Mock UploadFile for testing."""
        pdf_buffer = io.BytesIO(test_pdf_content)
        return UploadFile(file=pdf_buffer, filename="llama_stack_and_models.pdf")

    async def test_process_file_basic(self, processor: DoclingFileProcessor, upload_file: UploadFile):
        """Test basic file processing without chunking."""
        upload_file.file.seek(0)
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Verify response structure
        assert response.chunks is not None
        assert response.metadata is not None
        assert len(response.chunks) == 1  # Single chunk without chunking strategy

        # Verify metadata
        metadata = response.metadata
        assert metadata["processor"] == "docling"
        assert "processing_time_ms" in metadata
        assert isinstance(metadata["processing_time_ms"], int)
        assert metadata["processing_time_ms"] >= 0
        assert "page_count" in metadata
        assert metadata["page_count"] > 0
        assert metadata["extraction_method"] == "docling"
        assert "file_size_bytes" in metadata

        # Verify chunk content and metadata
        chunk = response.chunks[0]
        assert chunk.content is not None
        assert len(chunk.content.strip()) > 0
        assert chunk.chunk_id is not None
        assert chunk.chunk_metadata is not None
        assert chunk.chunk_metadata.content_token_count > 0
        uuid.UUID(chunk.chunk_metadata.document_id)  # Should be a valid UUID

    async def test_process_file_with_auto_chunking(self, processor: DoclingFileProcessor, upload_file: UploadFile):
        """Test file processing with auto chunking strategy."""
        upload_file.file.seek(0)
        chunking_strategy = VectorStoreChunkingStrategyAuto()
        response = await processor.process_file(file=upload_file, chunking_strategy=chunking_strategy)

        # Verify response structure
        assert response.chunks is not None
        assert len(response.chunks) >= 1

        # Collect chunk IDs to verify uniqueness
        chunk_ids: set[str] = set()

        for chunk in response.chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.chunk_id is not None
            assert chunk.chunk_id not in chunk_ids
            chunk_ids.add(chunk.chunk_id)

            assert chunk.chunk_metadata is not None
            assert chunk.chunk_metadata.content_token_count > 0
            uuid.UUID(chunk.chunk_metadata.document_id)
            assert chunk.chunk_metadata.chunk_window is not None

            assert "document_id" in chunk.metadata
            uuid.UUID(chunk.metadata["document_id"])
            assert chunk.metadata["filename"] == "llama_stack_and_models.pdf"

    async def test_process_file_with_static_chunking(self, processor: DoclingFileProcessor, upload_file: UploadFile):
        """Test file processing with static chunking strategy."""
        upload_file.file.seek(0)
        static_config = VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=256, chunk_overlap_tokens=25)
        chunking_strategy = VectorStoreChunkingStrategyStatic(static=static_config)
        response = await processor.process_file(file=upload_file, chunking_strategy=chunking_strategy)

        # Verify response structure
        assert response.chunks is not None
        assert len(response.chunks) > 1  # Should create multiple chunks

        # Collect chunk IDs to verify uniqueness
        chunk_ids: set[str] = set()

        for chunk in response.chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.chunk_id is not None
            assert chunk.chunk_id not in chunk_ids
            chunk_ids.add(chunk.chunk_id)
            assert chunk.chunk_metadata.content_token_count > 0
            uuid.UUID(chunk.chunk_metadata.document_id)

    async def test_input_validation(self, processor: DoclingFileProcessor):
        """Test input validation."""
        # Test no file or file_id provided
        with pytest.raises(ValueError, match="Either file or file_id must be provided"):
            await processor.process_file()

        # Test both file and file_id provided
        upload_file = UploadFile(file=io.BytesIO(b"test"), filename="test.pdf")
        with pytest.raises(ValueError, match="Cannot provide both file and file_id"):
            await processor.process_file(file=upload_file, file_id="test_id")

    async def test_metadata_extraction(self, processor: DoclingFileProcessor, upload_file: UploadFile):
        """Test metadata extraction."""
        upload_file.file.seek(0)
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        metadata = response.metadata
        assert "page_count" in metadata
        assert isinstance(metadata["page_count"], int)
        assert metadata["page_count"] > 0

        chunk = response.chunks[0]
        assert "filename" in chunk.metadata
        assert chunk.metadata["filename"] == "llama_stack_and_models.pdf"
        uuid.UUID(chunk.metadata["document_id"])

    async def test_chunk_id_uniqueness(self, processor: DoclingFileProcessor, upload_file: UploadFile):
        """Test chunk ID uniqueness across chunks."""
        upload_file.file.seek(0)
        chunking_strategy = VectorStoreChunkingStrategyAuto()
        response = await processor.process_file(file=upload_file, chunking_strategy=chunking_strategy)

        chunk_ids = [chunk.chunk_id for chunk in response.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

        for chunk in response.chunks:
            assert chunk.chunk_id is not None
            assert chunk.chunk_id != ""
            assert chunk.chunk_metadata.chunk_id == chunk.chunk_id

    async def test_max_page_count(self, processor: DoclingFileProcessor, upload_file: UploadFile):
        """Test that page count is reported and documents within max_page_count are processed."""
        upload_file.file.seek(0)
        response = await processor.process_file(file=upload_file, chunking_strategy=None)
        assert response.metadata["page_count"] == 1
        assert response.metadata["page_count"] <= processor.config.max_page_count
        assert len(response.chunks) == 1


class TestDoclingFileProcessorConfig:
    """Tests for Docling file processor configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = DoclingFileProcessorConfig()

        assert config.default_chunk_size_tokens >= 100
        assert config.default_chunk_overlap_tokens >= 0
        assert config.max_file_size_bytes == 100 * 1024 * 1024
        assert config.max_page_count == 100

    def test_config_validation(self):
        """Test configuration validation."""
        config = DoclingFileProcessorConfig(
            default_chunk_size_tokens=500,
            default_chunk_overlap_tokens=100,
            max_page_count=50,
        )
        assert config.default_chunk_size_tokens == 500
        assert config.default_chunk_overlap_tokens == 100
        assert config.max_page_count == 50

    def test_sample_run_config(self):
        """Test sample_run_config returns empty dict."""
        assert DoclingFileProcessorConfig.sample_run_config() == {}
