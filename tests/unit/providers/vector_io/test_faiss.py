# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.faiss.faiss import (
    FaissIndex,
    FaissVectorIOAdapter,
)
from llama_stack_api import Chunk, ChunkMetadata, EmbeddedChunk, Files, HealthStatus, QueryChunksResponse, VectorStore

# This test is a unit test for the FaissVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_faiss.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

FAISS_PROVIDER = "faiss"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def embedding_dimension():
    return 768


@pytest.fixture
def vector_store_id():
    return "test_vector_store"


@pytest.fixture
def sample_chunks():
    import time

    from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunk_id_1 = generate_chunk_id("mock-doc-1", "MOCK text content 1")
    chunk_id_2 = generate_chunk_id("mock-doc-2", "MOCK text content 1")

    return [
        Chunk(
            content="MOCK text content 1",
            chunk_id=chunk_id_1,
            metadata={"document_id": "mock-doc-1"},
            chunk_metadata=ChunkMetadata(
                chunk_id=chunk_id_1,
                document_id="mock-doc-1",
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=4,
            ),
        ),
        Chunk(
            content="MOCK text content 1",
            chunk_id=chunk_id_2,
            metadata={"document_id": "mock-doc-2"},
            chunk_metadata=ChunkMetadata(
                chunk_id=chunk_id_2,
                document_id="mock-doc-2",
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=4,
            ),
        ),
    ]


@pytest.fixture
def sample_embeddings(embedding_dimension):
    return np.random.rand(2, embedding_dimension).astype(np.float32)


@pytest.fixture
def mock_vector_store(vector_store_id, embedding_dimension) -> MagicMock:
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_vector_store.embedding_model = "mock_embedding_model"
    mock_vector_store.identifier = vector_store_id
    mock_vector_store.embedding_dimension = embedding_dimension
    return mock_vector_store


@pytest.fixture
def mock_files_api():
    mock_api = MagicMock(spec=Files)
    return mock_api


@pytest.fixture
def faiss_config():
    config = MagicMock(spec=FaissVectorIOConfig)
    config.kvstore = None
    return config


@pytest.fixture
async def faiss_index(embedding_dimension):
    index = await FaissIndex.create(dimension=embedding_dimension)
    yield index


async def test_faiss_query_vector_returns_infinity_when_query_and_embedding_are_identical(
    faiss_index, sample_chunks, sample_embeddings, embedding_dimension
):
    # Create EmbeddedChunk objects from chunks and embeddings
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            chunk_metadata=chunk.chunk_metadata,
            embedding=embedding.tolist(),
            embedding_model="test-embedding-model",
            embedding_dimension=embedding_dimension,
        )
        for chunk, embedding in zip(sample_chunks, sample_embeddings, strict=False)
    ]
    await faiss_index.add_chunks(embedded_chunks)
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)

    with patch.object(faiss_index.index, "search") as mock_search:
        mock_search.return_value = (np.array([[0.0, 0.1]]), np.array([[0, 1]]))

        response = await faiss_index.query_vector(embedding=query_embedding, k=2, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2

        assert response.scores[0] == float("inf")  # infinity (1.0 / 0.0)
        assert response.scores[1] == 10.0  # (1.0 / 0.1 = 10.0)

        assert response.chunks[0] == embedded_chunks[0]
        assert response.chunks[1] == embedded_chunks[1]


async def test_health_success():
    """Test that the health check returns OK status when faiss is working correctly."""
    # Create a fresh instance of FaissVectorIOAdapter for testing
    config = MagicMock()
    inference_api = MagicMock()
    files_api = MagicMock()

    with patch("llama_stack.providers.inline.vector_io.faiss.faiss.faiss.IndexFlatL2") as mock_index_flat:
        mock_index_flat.return_value = MagicMock()
        adapter = FaissVectorIOAdapter(config=config, inference_api=inference_api, files_api=files_api)

        # Calling the health method directly
        response = await adapter.health()

        # Verifying the response
        assert isinstance(response, dict)
        assert response["status"] == HealthStatus.OK
        assert "message" not in response

        # Verifying that IndexFlatL2 was called with the correct dimension
        mock_index_flat.assert_called_once_with(128)  # VECTOR_DIMENSION is 128


async def test_health_failure():
    """Test that the health check returns ERROR status when faiss encounters an error."""
    # Create a fresh instance of FaissVectorIOAdapter for testing
    config = MagicMock()
    inference_api = MagicMock()
    files_api = MagicMock()

    with patch("llama_stack.providers.inline.vector_io.faiss.faiss.faiss.IndexFlatL2") as mock_index_flat:
        mock_index_flat.side_effect = Exception("Test error")

        adapter = FaissVectorIOAdapter(config=config, inference_api=inference_api, files_api=files_api)

        # Calling the health method directly
        response = await adapter.health()

        # Verifying the response
        assert isinstance(response, dict)
        assert response["status"] == HealthStatus.ERROR
        assert response["message"] == "Health check failed: Test error"
