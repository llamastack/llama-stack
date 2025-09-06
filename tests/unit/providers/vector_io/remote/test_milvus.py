# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llama_stack.apis.vector_io import QueryChunksResponse

# Mock the entire pymilvus module
pymilvus_mock = MagicMock()
pymilvus_mock.DataType = MagicMock()
pymilvus_mock.MilvusClient = MagicMock
pymilvus_mock.RRFRanker = MagicMock
pymilvus_mock.WeightedRanker = MagicMock
pymilvus_mock.AnnSearchRequest = MagicMock

# Apply the mock before importing MilvusIndex
with patch.dict("sys.modules", {"pymilvus": pymilvus_mock}):
    from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusIndex

# This test is a unit test for the MilvusIndex class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/remote/test_milvus.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

MILVUS_PROVIDER = "milvus"


@pytest.fixture
async def mock_milvus_client() -> MagicMock:
    """Create a mock Milvus client with common method behaviors."""
    client = MagicMock()

    # Mock collection operations
    client.has_collection.return_value = False  # Initially no collection
    client.create_collection.return_value = None
    client.drop_collection.return_value = None

    # Mock insert operation
    client.insert.return_value = {"insert_count": 10}

    # Mock search operation - return mock results (data should be dict, not JSON string)
    client.search.return_value = [
        [
            {
                "id": 0,
                "distance": 0.1,
                "entity": {"chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}}},
            },
            {
                "id": 1,
                "distance": 0.2,
                "entity": {"chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}}},
            },
        ]
    ]

    # Mock query operation for keyword search (data should be dict, not JSON string)
    client.query.return_value = [
        {
            "chunk_id": "chunk1",
            "chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}},
            "score": 0.9,
        },
        {
            "chunk_id": "chunk2",
            "chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}},
            "score": 0.8,
        },
        {
            "chunk_id": "chunk3",
            "chunk_content": {"content": "mock chunk 3", "metadata": {"document_id": "doc3"}},
            "score": 0.7,
        },
    ]

    return client


@pytest.fixture
async def milvus_index(mock_milvus_client):
    """Create a MilvusIndex with mocked client."""
    index = MilvusIndex(client=mock_milvus_client, collection_name="test_collection")
    yield index
    # No real cleanup needed since we're using mocks


async def test_add_chunks(milvus_index, sample_chunks, sample_embeddings, mock_milvus_client):
    # Setup: collection doesn't exist initially, then exists after creation
    mock_milvus_client.has_collection.side_effect = [False, True]

    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Verify collection was created and data was inserted
    mock_milvus_client.create_collection.assert_called_once()
    mock_milvus_client.insert.assert_called_once()

    # Verify the insert call had the right number of chunks
    insert_call = mock_milvus_client.insert.call_args
    assert len(insert_call[1]["data"]) == len(sample_chunks)


async def test_query_chunks_vector(
    milvus_index, sample_chunks, sample_embeddings, embedding_dimension, mock_milvus_client
):
    # Setup: Add chunks first
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Test vector search
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    response = await milvus_index.query_vector(query_embedding, k=2, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    mock_milvus_client.search.assert_called_once()


async def test_query_chunks_keyword_search(milvus_index, sample_chunks, sample_embeddings, mock_milvus_client):
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Test keyword search
    query_string = "Sentence 5"
    response = await milvus_index.query_keyword(query_string=query_string, k=2, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


async def test_bm25_fallback_to_simple_search(milvus_index, sample_chunks, sample_embeddings, mock_milvus_client):
    """Test that when BM25 search fails, the system falls back to simple text search."""
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Force BM25 search to fail
    mock_milvus_client.search.side_effect = Exception("BM25 search not available")

    # Mock simple text search results
    mock_milvus_client.query.return_value = [
        {
            "chunk_id": "chunk1",
            "chunk_content": {"content": "Python programming language", "metadata": {"document_id": "doc1"}},
        },
        {
            "chunk_id": "chunk2",
            "chunk_content": {"content": "Machine learning algorithms", "metadata": {"document_id": "doc2"}},
        },
    ]

    # Test keyword search that should fall back to simple text search
    query_string = "Python"
    response = await milvus_index.query_keyword(query_string=query_string, k=3, score_threshold=0.0)

    # Verify response structure
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) > 0, "Fallback search should return results"

    # Verify that simple text search was used (query method called instead of search)
    mock_milvus_client.query.assert_called_once()
    mock_milvus_client.search.assert_called_once()  # Called once but failed

    # Verify the query uses parameterized filter with filter_params
    query_call_args = mock_milvus_client.query.call_args
    assert "filter" in query_call_args[1], "Query should include filter for text search"
    assert "filter_params" in query_call_args[1], "Query should use parameterized filter"
    assert query_call_args[1]["filter_params"]["content"] == "Python", "Filter params should contain the search term"

    # Verify all returned chunks have score 1.0 (simple binary scoring)
    assert all(score == 1.0 for score in response.scores), "Simple text search should use binary scoring"


async def test_delete_collection(milvus_index, mock_milvus_client):
    # Test collection deletion
    mock_milvus_client.has_collection.return_value = True

    await milvus_index.delete()

    mock_milvus_client.drop_collection.assert_called_once_with(collection_name=milvus_index.collection_name)


async def test_query_hybrid_search_rrf(
    milvus_index, sample_chunks, sample_embeddings, embedding_dimension, mock_milvus_client
):
    """Test hybrid search with RRF reranker."""
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Mock hybrid search results
    mock_milvus_client.hybrid_search.return_value = [
        [
            {
                "id": 0,
                "distance": 0.1,
                "entity": {"chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}}},
            },
            {
                "id": 1,
                "distance": 0.2,
                "entity": {"chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}}},
            },
        ]
    ]

    # Test hybrid search with RRF reranker
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    query_string = "test query"
    response = await milvus_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=2,
        score_threshold=0.0,
        reranker_type="rrf",
        reranker_params={"impact_factor": 60.0},
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    assert len(response.scores) == 2

    # Verify hybrid search was called with correct parameters
    mock_milvus_client.hybrid_search.assert_called_once()
    call_args = mock_milvus_client.hybrid_search.call_args

    # Check that the request contains both vector and BM25 search requests
    reqs = call_args[1]["reqs"]
    assert len(reqs) == 2
    assert reqs[0].anns_field == "vector"
    assert reqs[1].anns_field == "sparse"
    ranker = call_args[1]["ranker"]
    assert ranker is not None


async def test_query_hybrid_search_weighted(
    milvus_index, sample_chunks, sample_embeddings, embedding_dimension, mock_milvus_client
):
    """Test hybrid search with weighted reranker."""
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Mock hybrid search results
    mock_milvus_client.hybrid_search.return_value = [
        [
            {
                "id": 0,
                "distance": 0.1,
                "entity": {"chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}}},
            },
            {
                "id": 1,
                "distance": 0.2,
                "entity": {"chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}}},
            },
        ]
    ]

    # Test hybrid search with weighted reranker
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    query_string = "test query"
    response = await milvus_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=2,
        score_threshold=0.0,
        reranker_type="weighted",
        reranker_params={"alpha": 0.7},
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    assert len(response.scores) == 2

    # Verify hybrid search was called with correct parameters
    mock_milvus_client.hybrid_search.assert_called_once()
    call_args = mock_milvus_client.hybrid_search.call_args
    ranker = call_args[1]["ranker"]
    assert ranker is not None


async def test_query_hybrid_search_default_rrf(
    milvus_index, sample_chunks, sample_embeddings, embedding_dimension, mock_milvus_client
):
    """Test hybrid search with default RRF reranker (no reranker_type specified)."""
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Mock hybrid search results
    mock_milvus_client.hybrid_search.return_value = [
        [
            {
                "id": 0,
                "distance": 0.1,
                "entity": {"chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}}},
            },
        ]
    ]

    # Test hybrid search with default reranker (should be RRF)
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    query_string = "test query"
    response = await milvus_index.query_hybrid(
        embedding=query_embedding,
        query_string=query_string,
        k=1,
        score_threshold=0.0,
        reranker_type="unknown_type",  # Should default to RRF
        reranker_params=None,  # Should use default impact_factor
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 1

    # Verify hybrid search was called with RRF reranker
    mock_milvus_client.hybrid_search.assert_called_once()
    call_args = mock_milvus_client.hybrid_search.call_args
    ranker = call_args[1]["ranker"]
    assert ranker is not None


# TODO: Write tests for the MilvusVectorIOAdapter class.
