# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import time

import pytest
from openai import OpenAI

from llama_stack.apis.vector_io import Chunk
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

logger = logging.getLogger(__name__)


def skip_if_provider_doesnt_support_openai_vector_stores(client_with_models):
    vector_io_providers = [p for p in client_with_models.providers.list() if p.api == "vector_io"]
    for p in vector_io_providers:
        if p.provider_type in ["inline::faiss", "inline::sqlite-vec"]:
            return

    pytest.skip("OpenAI vector stores are not supported by any provider")


@pytest.fixture
def openai_client(client_with_models):
    base_url = f"{client_with_models.base_url}/v1/openai/v1"
    return OpenAI(base_url=base_url, api_key="fake")


@pytest.fixture(params=["openai_client", "llama_stack_client"])
def compat_client(request, client_with_models):
    if request.param == "openai_client" and isinstance(client_with_models, LlamaStackAsLibraryClient):
        pytest.skip("OpenAI client tests not supported with library client")
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def sample_chunks():
    return [
        Chunk(
            content="Python is a high-level programming language that emphasizes code readability and allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java.",
            metadata={"document_id": "doc1", "topic": "programming"},
        ),
        Chunk(
            content="Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed, using statistical techniques to give computer systems the ability to progressively improve performance on a specific task.",
            metadata={"document_id": "doc2", "topic": "ai"},
        ),
        Chunk(
            content="Data structures are fundamental to computer science because they provide organized ways to store and access data efficiently, enable faster processing of data through optimized algorithms, and form the building blocks for more complex software systems.",
            metadata={"document_id": "doc3", "topic": "computer_science"},
        ),
        Chunk(
            content="Neural networks are inspired by biological neural networks found in animal brains, using interconnected nodes called artificial neurons to process information through weighted connections that can be trained to recognize patterns and solve complex problems through iterative learning.",
            metadata={"document_id": "doc4", "topic": "ai"},
        ),
    ]


@pytest.fixture(scope="function")
def compat_client_with_empty_stores(compat_client):
    def clear_vector_stores():
        # List and delete all existing vector stores
        try:
            response = compat_client.vector_stores.list()
            for store in response.data:
                compat_client.vector_stores.delete(vector_store_id=store.id)
        except Exception:
            # If the API is not available or fails, just continue
            logger.warning("Failed to clear vector stores")
            pass

    clear_vector_stores()
    yield compat_client

    # Clean up after the test
    clear_vector_stores()


def test_openai_create_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test creating a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = client.vector_stores.create(
        name="test_vector_store", metadata={"purpose": "testing", "environment": "integration"}
    )

    assert vector_store is not None
    assert vector_store.name == "test_vector_store"
    assert vector_store.object == "vector_store"
    assert vector_store.status in ["completed", "in_progress"]
    assert vector_store.metadata["purpose"] == "testing"
    assert vector_store.metadata["environment"] == "integration"
    assert hasattr(vector_store, "id")
    assert hasattr(vector_store, "created_at")


def test_openai_list_vector_stores(compat_client_with_empty_stores, client_with_models):
    """Test listing vector stores using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a few vector stores
    store1 = client.vector_stores.create(name="store1", metadata={"type": "test"})
    store2 = client.vector_stores.create(name="store2", metadata={"type": "test"})

    # List vector stores
    response = client.vector_stores.list()

    assert response is not None
    assert hasattr(response, "data")
    assert len(response.data) >= 2

    # Check that our stores are in the list
    store_ids = [store.id for store in response.data]
    assert store1.id in store_ids
    assert store2.id in store_ids

    # Test pagination with limit
    limited_response = client.vector_stores.list(limit=1)
    assert len(limited_response.data) == 1


def test_openai_retrieve_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test retrieving a specific vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    created_store = client.vector_stores.create(name="retrieve_test_store", metadata={"purpose": "retrieval_test"})

    # Retrieve the store
    retrieved_store = client.vector_stores.retrieve(vector_store_id=created_store.id)

    assert retrieved_store is not None
    assert retrieved_store.id == created_store.id
    assert retrieved_store.name == "retrieve_test_store"
    assert retrieved_store.metadata["purpose"] == "retrieval_test"
    assert retrieved_store.object == "vector_store"


def test_openai_update_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test modifying a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    created_store = client.vector_stores.create(name="original_name", metadata={"version": "1.0"})
    time.sleep(1)
    # Modify the store
    modified_store = client.vector_stores.update(
        vector_store_id=created_store.id, name="modified_name", metadata={"version": "1.1", "updated": "true"}
    )

    assert modified_store is not None
    assert modified_store.id == created_store.id
    assert modified_store.name == "modified_name"
    assert modified_store.metadata["version"] == "1.1"
    assert modified_store.metadata["updated"] == "true"
    # last_active_at should be updated
    assert modified_store.last_active_at > created_store.last_active_at


def test_openai_delete_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test deleting a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    created_store = client.vector_stores.create(name="delete_test_store", metadata={"purpose": "deletion_test"})

    # Delete the store
    delete_response = client.vector_stores.delete(vector_store_id=created_store.id)

    assert delete_response is not None
    assert delete_response.id == created_store.id
    assert delete_response.deleted is True
    assert delete_response.object == "vector_store.deleted"

    # Verify the store is deleted - attempting to retrieve should fail
    with pytest.raises(Exception):  # noqa: B017
        client.vector_stores.retrieve(vector_store_id=created_store.id)


def test_openai_vector_store_search_empty(compat_client_with_empty_stores, client_with_models):
    """Test searching an empty vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = client.vector_stores.create(name="search_test_store", metadata={"purpose": "search_testing"})

    # Search the empty store
    search_response = client.vector_stores.search(
        vector_store_id=vector_store.id, query="test query", max_num_results=5
    )

    assert search_response is not None
    assert hasattr(search_response, "data")
    assert len(search_response.data) == 0  # Empty store should return no results
    assert search_response.search_query == "test query"
    assert search_response.has_more is False


def test_openai_vector_store_with_chunks(compat_client_with_empty_stores, client_with_models, sample_chunks):
    """Test vector store functionality with actual chunks using both OpenAI and native APIs."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store using OpenAI API
    vector_store = compat_client.vector_stores.create(name="chunks_test_store", metadata={"purpose": "chunks_testing"})

    # Insert chunks using the native LlamaStack API (since OpenAI API doesn't have direct chunk insertion)
    llama_client.vector_io.insert(
        vector_db_id=vector_store.id,
        chunks=sample_chunks,
    )

    # Search using OpenAI API
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id, query="What is Python programming language?", max_num_results=3
    )
    assert search_response is not None
    assert len(search_response.data) > 0

    # The top result should be about Python (doc1)
    top_result = search_response.data[0]
    top_content = top_result.content[0].text
    assert "python" in top_content.lower() or "programming" in top_content.lower()
    assert top_result.attributes["document_id"] == "doc1"

    # Test filtering by metadata
    filtered_search = compat_client.vector_stores.search(
        vector_store_id=vector_store.id, query="artificial intelligence", filters={"topic": "ai"}, max_num_results=5
    )

    assert filtered_search is not None
    # All results should have topic "ai"
    for result in filtered_search.data:
        assert result.attributes["topic"] == "ai"


@pytest.mark.parametrize(
    "test_case",
    [
        ("What makes Python different from other languages?", "doc1", "programming"),
        ("How do systems learn automatically?", "doc2", "ai"),
        ("Why are data structures important?", "doc3", "computer_science"),
        ("What inspires neural networks?", "doc4", "ai"),
    ],
)
def test_openai_vector_store_search_relevance(
    compat_client_with_empty_stores, client_with_models, sample_chunks, test_case
):
    """Test that OpenAI vector store search returns relevant results for different queries."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    query, expected_doc_id, expected_topic = test_case

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name=f"relevance_test_{expected_doc_id}", metadata={"purpose": "relevance_testing"}
    )

    # Insert chunks using native API
    llama_client.vector_io.insert(
        vector_db_id=vector_store.id,
        chunks=sample_chunks,
    )

    # Search using OpenAI API
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id, query=query, max_num_results=4
    )

    assert search_response is not None
    assert len(search_response.data) > 0

    # The top result should match the expected document
    top_result = search_response.data[0]

    assert top_result.attributes["document_id"] == expected_doc_id
    assert top_result.attributes["topic"] == expected_topic

    # Verify score is included and reasonable
    assert isinstance(top_result.score, int | float)
    assert top_result.score > 0


def test_openai_vector_store_search_with_ranking_options(
    compat_client_with_empty_stores, client_with_models, sample_chunks
):
    """Test OpenAI vector store search with ranking options."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="ranking_test_store", metadata={"purpose": "ranking_testing"}
    )

    # Insert chunks
    llama_client.vector_io.insert(
        vector_db_id=vector_store.id,
        chunks=sample_chunks,
    )

    # Search with ranking options
    threshold = 0.1
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id,
        query="machine learning and artificial intelligence",
        max_num_results=3,
        ranking_options={"score_threshold": threshold},
    )

    assert search_response is not None
    assert len(search_response.data) > 0

    # All results should meet the score threshold
    for result in search_response.data:
        assert result.score >= threshold


def test_openai_vector_store_search_with_high_score_filter(
    compat_client_with_empty_stores, client_with_models, sample_chunks
):
    """Test that searching with text very similar to a document and high score threshold returns only that document."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="high_score_filter_test", metadata={"purpose": "high_score_filtering"}
    )

    # Insert chunks
    llama_client.vector_io.insert(
        vector_db_id=vector_store.id,
        chunks=sample_chunks,
    )

    # Query with text very similar to the Python document (doc1)
    # This should match very closely to the first sample chunk about Python
    query = "Python is a high-level programming language with code readability and fewer lines than C++ or Java"

    # picking up thrshold to be slightly higher than the second result
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id,
        query=query,
        max_num_results=3,
    )
    assert len(search_response.data) > 1, "Expected more than one result"
    threshold = search_response.data[1].score + 0.0001

    # we expect only one result with the requested threshold
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id,
        query=query,
        max_num_results=10,  # Allow more results but expect filtering
        ranking_options={"score_threshold": threshold},
    )

    # With high threshold and similar query, we should get only the Python document
    assert len(search_response.data) == 1, "Expected only one result with high threshold"

    # The top result should be the Python document (doc1)
    top_result = search_response.data[0]
    assert top_result.attributes["document_id"] == "doc1"
    assert top_result.attributes["topic"] == "programming"
    assert top_result.score >= threshold

    # Verify the content contains Python-related terms
    top_content = top_result.content[0].text
    assert "python" in top_content.lower() or "programming" in top_content.lower()


def test_openai_vector_store_search_with_max_num_results(
    compat_client_with_empty_stores, client_with_models, sample_chunks
):
    """Test OpenAI vector store search with max_num_results."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="max_num_results_test_store", metadata={"purpose": "max_num_results_testing"}
    )

    # Insert chunks
    llama_client.vector_io.insert(
        vector_db_id=vector_store.id,
        chunks=sample_chunks,
    )

    # Search with max_num_results
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id,
        query="machine learning and artificial intelligence",
        max_num_results=2,
    )

    assert search_response is not None
    assert len(search_response.data) == 2
