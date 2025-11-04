# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from llama_stack.apis.inference.inference import (
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
)
from llama_stack.apis.tools import RAGDocument
from llama_stack.apis.vector_io import Chunk
from llama_stack.providers.utils.memory.vector_store import (
    URL,
    VectorStoreWithIndex,
    _validate_embedding,
    content_from_doc,
    make_overlapped_chunks,
)
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

DUMMY_PDF_PATH = Path(os.path.abspath(__file__)).parent / "fixtures" / "dummy.pdf"
# Depending on the machine, this can get parsed a couple of ways
DUMMY_PDF_TEXT_CHOICES = ["Dummy PDF file", "Dumm y PDF file"]


def read_file(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()


def data_url_from_file(file_path: str) -> str:
    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


class TestChunk:
    def test_chunk(self):
        chunk = Chunk(
            content="Example chunk content",
            chunk_id=generate_chunk_id("test-doc", "Example chunk content"),
            metadata={"key": "value"},
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk.content == "Example chunk content"
        assert chunk.metadata == {"key": "value"}
        assert chunk.embedding == [0.1, 0.2, 0.3]

        chunk_no_embedding = Chunk(
            content="Example chunk content",
            chunk_id=generate_chunk_id("test-doc", "Example chunk content"),
            metadata={"key": "value"},
        )
        assert chunk_no_embedding.embedding is None


class TestValidateEmbedding:
    def test_valid_list_embeddings(self):
        _validate_embedding([0.1, 0.2, 0.3], 0, 3)
        _validate_embedding([1, 2, 3], 1, 3)
        _validate_embedding([0.1, 2, 3.5], 2, 3)

    def test_valid_numpy_embeddings(self):
        _validate_embedding(np.array([0.1, 0.2, 0.3], dtype=np.float32), 0, 3)
        _validate_embedding(np.array([0.1, 0.2, 0.3], dtype=np.float64), 1, 3)
        _validate_embedding(np.array([1, 2, 3], dtype=np.int32), 2, 3)
        _validate_embedding(np.array([1, 2, 3], dtype=np.int64), 3, 3)

    def test_invalid_embedding_type(self):
        error_msg = "must be a list or numpy array"

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding("not a list", 0, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding(None, 1, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding(42, 2, 3)

    def test_non_numeric_values(self):
        error_msg = "contains non-numeric values"

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding([0.1, "string", 0.3], 0, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding([0.1, None, 0.3], 1, 3)

        with pytest.raises(ValueError, match=error_msg):
            _validate_embedding([1, {}, 3], 2, 3)

    def test_wrong_dimension(self):
        with pytest.raises(ValueError, match="has dimension 4, expected 3"):
            _validate_embedding([0.1, 0.2, 0.3, 0.4], 0, 3)

        with pytest.raises(ValueError, match="has dimension 2, expected 3"):
            _validate_embedding([0.1, 0.2], 1, 3)

        with pytest.raises(ValueError, match="has dimension 0, expected 3"):
            _validate_embedding([], 2, 3)


class TestVectorStore:
    async def test_returns_content_from_pdf_data_uri(self):
        data_uri = data_url_from_file(DUMMY_PDF_PATH)
        doc = RAGDocument(
            document_id="dummy",
            content=data_uri,
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content in DUMMY_PDF_TEXT_CHOICES

    @pytest.mark.allow_network
    async def test_downloads_pdf_and_returns_content(self):
        # Using GitHub to host the PDF file
        url = "https://raw.githubusercontent.com/meta-llama/llama-stack/da035d69cfca915318eaf485770a467ca3c2a238/llama_stack/providers/tests/memory/fixtures/dummy.pdf"
        doc = RAGDocument(
            document_id="dummy",
            content=url,
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content in DUMMY_PDF_TEXT_CHOICES

    @pytest.mark.allow_network
    async def test_downloads_pdf_and_returns_content_with_url_object(self):
        # Using GitHub to host the PDF file
        url = "https://raw.githubusercontent.com/meta-llama/llama-stack/da035d69cfca915318eaf485770a467ca3c2a238/llama_stack/providers/tests/memory/fixtures/dummy.pdf"
        doc = RAGDocument(
            document_id="dummy",
            content=URL(
                uri=url,
            ),
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content in DUMMY_PDF_TEXT_CHOICES

    @pytest.mark.parametrize(
        "window_len, overlap_len, expected_chunks",
        [
            (5, 2, 4),  # Create 4 chunks with window of 5 and overlap of 2
            (4, 1, 4),  # Create 4 chunks with window of 4 and overlap of 1
        ],
    )
    def test_make_overlapped_chunks(self, window_len, overlap_len, expected_chunks):
        document_id = "test_doc_123"
        text = "This is a sample document for testing the chunking behavior"
        original_metadata = {"source": "test", "date": "2023-01-01", "author": "llama"}
        len_metadata_tokens = 24  # specific to the metadata above

        chunks = make_overlapped_chunks(document_id, text, window_len, overlap_len, original_metadata)

        assert len(chunks) == expected_chunks

        # Check that each chunk has the right metadata
        for chunk in chunks:
            # Original metadata should be preserved
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["date"] == "2023-01-01"
            assert chunk.metadata["author"] == "llama"

            # New metadata should be added
            assert chunk.metadata["document_id"] == document_id
            assert "token_count" in chunk.metadata
            assert isinstance(chunk.metadata["token_count"], int)
            assert chunk.metadata["token_count"] > 0
            assert chunk.metadata["metadata_token_count"] == len_metadata_tokens

    def test_raise_overlapped_chunks_metadata_serialization_error(self):
        document_id = "test_doc_ex"
        text = "Some text"
        window_len = 5
        overlap_len = 2

        class BadMetadata:
            def __repr__(self):
                raise TypeError("Cannot convert to string")

        problematic_metadata = {"bad_metadata_example": BadMetadata()}

        with pytest.raises(ValueError) as excinfo:
            make_overlapped_chunks(document_id, text, window_len, overlap_len, problematic_metadata)

        assert str(excinfo.value) == "Failed to serialize metadata to string"
        assert isinstance(excinfo.value.__cause__, TypeError)
        assert str(excinfo.value.__cause__) == "Cannot convert to string"


class TestVectorStoreWithIndex:
    async def test_insert_chunks_without_embeddings(self):
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_model = "test-model without embeddings"
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()

        vector_store_with_index = VectorStoreWithIndex(
            vector_store=mock_vector_store, index=mock_index, inference_api=mock_inference_api
        )

        chunks = [
            Chunk(content="Test 1", chunk_id=generate_chunk_id("test-doc", "Test 1"), embedding=None, metadata={}),
            Chunk(content="Test 2", chunk_id=generate_chunk_id("test-doc", "Test 2"), embedding=None, metadata={}),
        ]

        mock_inference_api.openai_embeddings.return_value.data = [
            OpenAIEmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
            OpenAIEmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
        ]

        await vector_store_with_index.insert_chunks(chunks)

        # Verify openai_embeddings was called with correct params
        mock_inference_api.openai_embeddings.assert_called_once()
        call_args = mock_inference_api.openai_embeddings.call_args[0]
        assert len(call_args) == 1
        params = call_args[0]
        assert isinstance(params, OpenAIEmbeddingsRequestWithExtraBody)
        assert params.model == "test-model without embeddings"
        assert params.input == ["Test 1", "Test 2"]
        mock_index.add_chunks.assert_called_once()
        args = mock_index.add_chunks.call_args[0]
        assert args[0] == chunks
        assert np.array_equal(args[1], np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))

    async def test_insert_chunks_with_valid_embeddings(self):
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_model = "test-model with embeddings"
        mock_vector_store.embedding_dimension = 3
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()

        vector_store_with_index = VectorStoreWithIndex(
            vector_store=mock_vector_store, index=mock_index, inference_api=mock_inference_api
        )

        chunks = [
            Chunk(
                content="Test 1",
                chunk_id=generate_chunk_id("test-doc", "Test 1"),
                embedding=[0.1, 0.2, 0.3],
                metadata={},
            ),
            Chunk(
                content="Test 2",
                chunk_id=generate_chunk_id("test-doc", "Test 2"),
                embedding=[0.4, 0.5, 0.6],
                metadata={},
            ),
        ]

        await vector_store_with_index.insert_chunks(chunks)

        mock_inference_api.openai_embeddings.assert_not_called()
        mock_index.add_chunks.assert_called_once()
        args = mock_index.add_chunks.call_args[0]
        assert args[0] == chunks
        assert np.array_equal(args[1], np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))

    async def test_insert_chunks_with_invalid_embeddings(self):
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_dimension = 3
        mock_vector_store.embedding_model = "test-model with invalid embeddings"
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()

        vector_store_with_index = VectorStoreWithIndex(
            vector_store=mock_vector_store, index=mock_index, inference_api=mock_inference_api
        )

        # Verify Chunk raises ValueError for invalid embedding type
        with pytest.raises(ValueError, match="Input should be a valid list"):
            Chunk(
                content="Test 1",
                chunk_id=generate_chunk_id("test-doc", "Test 1"),
                embedding="invalid_type",
                metadata={},
            )

        # Verify Chunk raises ValueError for invalid embedding type in insert_chunks (i.e., Chunk errors before insert_chunks is called)
        with pytest.raises(ValueError, match="Input should be a valid list"):
            await vector_store_with_index.insert_chunks(
                [
                    Chunk(
                        content="Test 1", chunk_id=generate_chunk_id("test-doc", "Test 1"), embedding=None, metadata={}
                    ),
                    Chunk(
                        content="Test 2",
                        chunk_id=generate_chunk_id("test-doc", "Test 2"),
                        embedding="invalid_type",
                        metadata={},
                    ),
                ]
            )

        # Verify Chunk raises ValueError for invalid embedding element type in insert_chunks (i.e., Chunk errors before insert_chunks is called)
        with pytest.raises(ValueError, match=" Input should be a valid number, unable to parse string as a number "):
            await vector_store_with_index.insert_chunks(
                Chunk(
                    content="Test 1",
                    chunk_id=generate_chunk_id("test-doc", "Test 1"),
                    embedding=[0.1, "string", 0.3],
                    metadata={},
                )
            )

        chunks_wrong_dim = [
            Chunk(
                content="Test 1",
                chunk_id=generate_chunk_id("test-doc", "Test 1"),
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={},
            ),
        ]
        with pytest.raises(ValueError, match="has dimension 4, expected 3"):
            await vector_store_with_index.insert_chunks(chunks_wrong_dim)

        mock_inference_api.openai_embeddings.assert_not_called()
        mock_index.add_chunks.assert_not_called()

    async def test_insert_chunks_with_partially_precomputed_embeddings(self):
        mock_vector_store = MagicMock()
        mock_vector_store.embedding_model = "test-model with partial embeddings"
        mock_vector_store.embedding_dimension = 3
        mock_index = AsyncMock()
        mock_inference_api = AsyncMock()

        vector_store_with_index = VectorStoreWithIndex(
            vector_store=mock_vector_store, index=mock_index, inference_api=mock_inference_api
        )

        chunks = [
            Chunk(content="Test 1", chunk_id=generate_chunk_id("test-doc", "Test 1"), embedding=None, metadata={}),
            Chunk(
                content="Test 2",
                chunk_id=generate_chunk_id("test-doc", "Test 2"),
                embedding=[0.2, 0.2, 0.2],
                metadata={},
            ),
            Chunk(content="Test 3", chunk_id=generate_chunk_id("test-doc", "Test 3"), embedding=None, metadata={}),
        ]

        mock_inference_api.openai_embeddings.return_value.data = [
            OpenAIEmbeddingData(embedding=[0.1, 0.1, 0.1], index=0),
            OpenAIEmbeddingData(embedding=[0.3, 0.3, 0.3], index=1),
        ]

        await vector_store_with_index.insert_chunks(chunks)

        # Verify openai_embeddings was called with correct params
        mock_inference_api.openai_embeddings.assert_called_once()
        call_args = mock_inference_api.openai_embeddings.call_args[0]
        assert len(call_args) == 1
        params = call_args[0]
        assert isinstance(params, OpenAIEmbeddingsRequestWithExtraBody)
        assert params.model == "test-model with partial embeddings"
        assert params.input == ["Test 1", "Test 3"]
        mock_index.add_chunks.assert_called_once()
        args = mock_index.add_chunks.call_args[0]
        assert len(args[0]) == 3
        assert np.array_equal(args[1], np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], dtype=np.float32))
