# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from llama_stack.apis.vector_io import QueryChunksResponse

# Mock the Weaviate client
weaviate_mock = MagicMock()

# Apply the mock before importing WeaviateIndex
with patch.dict("sys.modules", {"weaviate": weaviate_mock}):
    from llama_stack.providers.remote.vector_io.weaviate.weaviate import WeaviateIndex

# This test is a unit test for the WeaviateIndex class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/remote/test_weaviate.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

WEAVIATE_PROVIDER = "weaviate"


@pytest.fixture
async def mock_weaviate_client() -> MagicMock:
    """Create a mock Weaviate client with common method behaviors."""
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Mock collection data operations
    mock_collection.data.insert_many.return_value = None
    mock_collection.data.delete_many.return_value = None

    # Mock collection search operations
    mock_collection.query.near_vector.return_value = None
    mock_collection.query.bm25.return_value = None
    mock_collection.query.hybrid.return_value = None
    mock_results = MagicMock()
    mock_results.objects = [MagicMock(), MagicMock()]
    mock_collection.query.near_vector.return_value = mock_results

    # Mock client collection operations
    mock_client.collections.get.return_value = mock_collection
    mock_client.collections.exists.return_value = True
    mock_client.collections.delete.return_value = None

    # Mock client close operation
    mock_client.close.return_value = None

    return mock_client


@pytest.fixture
async def weaviate_index(mock_weaviate_client):
    """Create a WeaviateIndex with mocked client."""
    index = WeaviateIndex(client=mock_weaviate_client, collection_name="Testcollection")
    yield index
    # No real cleanup needed since we're using mocks


async def test_add_chunks(weaviate_index, sample_chunks, sample_embeddings, mock_weaviate_client):
    # Setup: Add chunks first
    await weaviate_index.add_chunks(sample_chunks, sample_embeddings)

    # Assert
    mock_weaviate_client.collections.get.assert_called_once_with("Testcollection")
    mock_weaviate_client.collections.get.return_value.data.insert_many.assert_called_once()

    # Verify the insert call had the right number of chunks
    data_objects, _ = mock_weaviate_client.collections.get.return_value.data.insert_many.call_args
    assert len(data_objects[0]) == len(sample_chunks)


async def test_query_chunks_vector(
    weaviate_index, sample_chunks, sample_embeddings, embedding_dimension, mock_weaviate_client
):
    # Setup: Add chunks first
    await weaviate_index.add_chunks(sample_chunks, sample_embeddings)

    # Create mock objects that match Weaviate's response structure
    mock_objects = []
    for i, chunk in enumerate(sample_chunks[:2]):  # Return first 2 chunks
        mock_obj = MagicMock()
        mock_obj.properties = {"chunk_content": chunk.model_dump_json()}
        mock_obj.metadata.distance = 0.1 + i * 0.1  # Mock distances
        mock_objects.append(mock_obj)

    mock_results = MagicMock()
    mock_results.objects = mock_objects
    mock_weaviate_client.collections.get.return_value.query.near_vector.return_value = mock_results

    # Test vector search
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    response = await weaviate_index.query_vector(query_embedding, k=2, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    assert len(response.scores) == 2
    mock_weaviate_client.collections.get.return_value.query.near_vector.assert_called_once_with(
        near_vector=query_embedding.tolist(),
        limit=2,
        return_metadata=ANY,
    )


async def test_query_chunks_keyword_search(weaviate_index, sample_chunks, sample_embeddings, mock_weaviate_client):
    await weaviate_index.add_chunks(sample_chunks, sample_embeddings)

    # Find chunks that contain "Sentence 5"
    matching_chunks = [chunk for chunk in sample_chunks if "Sentence 5" in chunk.content]

    # Create mock objects that match Weaviate's BM25 response structure
    # Return the first 2 matching chunks
    mock_objects = []
    for i, chunk in enumerate(matching_chunks[:2]):
        mock_obj = MagicMock()
        mock_obj.properties = {"chunk_content": chunk.model_dump_json()}
        mock_obj.metadata.score = 0.9 - i * 0.1
        mock_objects.append(mock_obj)

    mock_results = MagicMock()
    mock_results.objects = mock_objects
    mock_weaviate_client.collections.get.return_value.query.bm25.return_value = mock_results

    # Test keyword search
    query_string = "Sentence 5"
    response = await weaviate_index.query_keyword(query_string=query_string, k=2, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    assert len(response.scores) == 2
    # Verify that the returned chunks contain the search term
    assert all("Sentence 5" in chunk.content for chunk in response.chunks)
    mock_weaviate_client.collections.get.return_value.query.bm25.assert_called_once_with(
        query=query_string,
        limit=2,
        return_metadata=ANY,
    )


async def test_delete_collection(weaviate_index, mock_weaviate_client):
    # Test collection deletion (when chunk_ids is None, it deletes the entire collection)
    mock_weaviate_client.collections.exists.return_value = True

    await weaviate_index.delete()

    mock_weaviate_client.collections.exists.assert_called_once_with("Testcollection")
    mock_weaviate_client.collections.delete.assert_called_once_with("Testcollection")


async def test_delete_chunks(weaviate_index, mock_weaviate_client):
    # Test deleting specific chunks using ChunkForDeletion objects
    from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion

    chunks_for_deletion = [
        ChunkForDeletion(chunk_id="chunk-1", document_id="doc-1"),
        ChunkForDeletion(chunk_id="chunk-2", document_id="doc-1"),
        ChunkForDeletion(chunk_id="chunk-3", document_id="doc-2"),
    ]

    await weaviate_index.delete_chunks(chunks_for_deletion)

    mock_weaviate_client.collections.get.assert_called_once_with("Testcollection")
    mock_weaviate_client.collections.get.return_value.data.delete_many.assert_called_once()


async def test_query_hybrid_rrf(
    weaviate_index, sample_chunks, sample_embeddings, embedding_dimension, mock_weaviate_client
):
    # Test hybrid search with RRF reranking
    from weaviate.classes.query import HybridFusion

    from llama_stack.providers.utils.memory.vector_store import RERANKER_TYPE_RRF

    await weaviate_index.add_chunks(sample_chunks, sample_embeddings)

    # Find chunks that contain "Sentence 5"
    matching_chunks = [chunk for chunk in sample_chunks if "Sentence 5" in chunk.content]

    # Create mock objects for hybrid search response
    mock_objects = []
    for i, chunk in enumerate(matching_chunks[:2]):
        mock_obj = MagicMock()
        mock_obj.properties = {"chunk_content": chunk.model_dump_json()}
        mock_obj.metadata.score = 0.85 + i * 0.05
        mock_objects.append(mock_obj)

    mock_results = MagicMock()
    mock_results.objects = mock_objects
    mock_weaviate_client.collections.get.return_value.query.hybrid.return_value = mock_results

    # Test hybrid search with RRF
    query_string = "Sentence 5"
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    response = await weaviate_index.query_hybrid(
        embedding=query_embedding, query_string=query_string, k=2, score_threshold=0.0, reranker_type=RERANKER_TYPE_RRF
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    assert len(response.scores) == 2
    assert all("Sentence 5" in chunk.content for chunk in response.chunks)

    # Verify the hybrid method was called with correct parameters
    mock_weaviate_client.collections.get.return_value.query.hybrid.assert_called_once_with(
        query=query_string,
        alpha=0.5,
        vector=query_embedding.tolist(),
        limit=2,
        fusion_type=HybridFusion.RANKED,
        return_metadata=ANY,
    )


async def test_query_hybrid_normalized(
    weaviate_index, sample_chunks, sample_embeddings, embedding_dimension, mock_weaviate_client
):
    from weaviate.classes.query import HybridFusion

    await weaviate_index.add_chunks(sample_chunks, sample_embeddings)

    # Find chunks that contain "Sentence 3" for different results
    matching_chunks = [chunk for chunk in sample_chunks if "Sentence 3" in chunk.content]

    # Create mock objects for hybrid search response
    mock_objects = []
    for i, chunk in enumerate(matching_chunks[:2]):
        mock_obj = MagicMock()
        mock_obj.properties = {"chunk_content": chunk.model_dump_json()}
        mock_obj.metadata.score = 0.75 + i * 0.1  # Mock hybrid scores
        mock_objects.append(mock_obj)

    mock_results = MagicMock()
    mock_results.objects = mock_objects
    mock_weaviate_client.collections.get.return_value.query.hybrid.return_value = mock_results

    # Test hybrid search with normalized reranking
    query_string = "Sentence 3"
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    response = await weaviate_index.query_hybrid(
        embedding=query_embedding, query_string=query_string, k=2, score_threshold=0.0, reranker_type="normalized"
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    assert len(response.scores) == 2
    assert all("Sentence 3" in chunk.content for chunk in response.chunks)

    # Verify the hybrid method was called with correct parameters
    mock_weaviate_client.collections.get.return_value.query.hybrid.assert_called_once_with(
        query=query_string,
        alpha=0.5,
        vector=query_embedding.tolist(),
        limit=2,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=ANY,
    )


# TODO: Write tests for the WeaviateVectorIOAdapter class.
