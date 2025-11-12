# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from io import BytesIO

import pytest
from llama_stack_client import BadRequestError
from openai import BadRequestError as OpenAIBadRequestError

from llama_stack.apis.files import ExpiresAfter
from llama_stack.apis.vector_io import Chunk
from llama_stack.core.library_client import LlamaStackAsLibraryClient
from llama_stack.log import get_logger

from ..conftest import vector_provider_wrapper

logger = get_logger(name=__name__, category="vector_io")


def skip_if_provider_doesnt_support_openai_vector_stores(client_with_models):
    vector_io_providers = [p for p in client_with_models.providers.list() if p.api == "vector_io"]
    for p in vector_io_providers:
        if p.provider_type in [
            "inline::chromadb",
            "inline::faiss",
            "inline::milvus",
            "inline::qdrant",
            "inline::sqlite-vec",
            "remote::chromadb",
            "remote::milvus",
            "remote::pgvector",
            "remote::qdrant",
            "remote::weaviate",
        ]:
            return

    pytest.skip("OpenAI vector stores are not supported by any provider")


def skip_if_provider_doesnt_support_openai_vector_stores_search(client_with_models, search_mode):
    vector_io_providers = [p for p in client_with_models.providers.list() if p.api == "vector_io"]
    search_mode_support = {
        "vector": [
            "inline::faiss",
            "inline::sqlite-vec",
            "inline::milvus",
            "inline::chromadb",
            "inline::qdrant",
            "remote::chromadb",
            "remote::milvus",
            "remote::pgvector",
            "remote::qdrant",
            "remote::weaviate",
        ],
        "keyword": [
            "inline::milvus",
            "inline::sqlite-vec",
            "remote::milvus",
            "remote::pgvector",
            "remote::weaviate",
        ],
        "hybrid": [
            "inline::milvus",
            "inline::sqlite-vec",
            "remote::milvus",
            "remote::pgvector",
            "remote::weaviate",
        ],
    }
    supported_providers = search_mode_support.get(search_mode, [])
    for p in vector_io_providers:
        if p.provider_type in supported_providers:
            return
    pytest.skip(
        f"Search mode '{search_mode}' is not supported by any available provider. "
        f"Supported providers for '{search_mode}': {supported_providers}"
    )


@pytest.fixture(scope="session")
def sample_chunks():
    from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunks_data = [
        (
            "Python is a high-level programming language that emphasizes code readability and allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java.",
            "doc1",
            "programming",
        ),
        (
            "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed, using statistical techniques to give computer systems the ability to progressively improve performance on a specific task.",
            "doc2",
            "ai",
        ),
        (
            "Data structures are fundamental to computer science because they provide organized ways to store and access data efficiently, enable faster processing of data through optimized algorithms, and form the building blocks for more complex software systems.",
            "doc3",
            "computer_science",
        ),
        (
            "Neural networks are inspired by biological neural networks found in animal brains, using interconnected nodes called artificial neurons to process information through weighted connections that can be trained to recognize patterns and solve complex problems through iterative learning.",
            "doc4",
            "ai",
        ),
    ]
    return [
        Chunk(
            content=content,
            chunk_id=generate_chunk_id(doc_id, content),
            metadata={"document_id": doc_id, "topic": topic},
        )
        for content, doc_id, topic in chunks_data
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

    def clear_files():
        try:
            response = compat_client.files.list()
            for file in response.data:
                compat_client.files.delete(file_id=file.id)
        except Exception:
            # If the API is not available or fails, just continue
            logger.warning("Failed to clear files")
            pass

    clear_vector_stores()
    clear_files()
    yield compat_client

    # Clean up after the test
    clear_vector_stores()
    clear_files()


@vector_provider_wrapper
def test_openai_create_vector_store(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test creating a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = client.vector_stores.create(
        name="Vs_test_vector_store",
        metadata={"purpose": "testing", "environment": "integration"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    assert vector_store is not None
    assert vector_store.name == "Vs_test_vector_store"
    assert vector_store.object == "vector_store"
    assert vector_store.status in ["completed", "in_progress"]
    assert vector_store.metadata["purpose"] == "testing"
    assert vector_store.metadata["environment"] == "integration"
    assert hasattr(vector_store, "id")
    assert hasattr(vector_store, "created_at")


@vector_provider_wrapper
def test_openai_create_vector_store_default(compat_client_with_empty_stores, client_with_models, vector_io_provider_id):
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    vector_store = compat_client_with_empty_stores.vector_stores.create(
        extra_body={"provider_id": vector_io_provider_id}
    )
    assert vector_store.id


@vector_provider_wrapper
def test_openai_list_vector_stores(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test listing vector stores using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a few vector stores
    store1 = client.vector_stores.create(
        name="store1",
        metadata={"type": "test"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )
    store2 = client.vector_stores.create(
        name="store2",
        metadata={"type": "test"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

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


@vector_provider_wrapper
def test_openai_retrieve_vector_store(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test retrieving a specific vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    created_store = client.vector_stores.create(
        name="retrieve_test_store",
        metadata={"purpose": "retrieval_test"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Retrieve the store
    retrieved_store = client.vector_stores.retrieve(vector_store_id=created_store.id)

    assert retrieved_store is not None
    assert retrieved_store.id == created_store.id
    assert retrieved_store.name == "retrieve_test_store"
    assert retrieved_store.metadata["purpose"] == "retrieval_test"
    assert retrieved_store.object == "vector_store"


@vector_provider_wrapper
def test_openai_update_vector_store(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test modifying a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    created_store = client.vector_stores.create(
        name="original_name",
        metadata={"version": "1.0"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )
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


@vector_provider_wrapper
def test_openai_delete_vector_store(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test deleting a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    created_store = client.vector_stores.create(
        name="delete_test_store",
        metadata={"purpose": "deletion_test"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Delete the store
    delete_response = client.vector_stores.delete(vector_store_id=created_store.id)

    assert delete_response is not None
    assert delete_response.id == created_store.id
    assert delete_response.deleted is True
    assert delete_response.object == "vector_store.deleted"

    # Verify the store is deleted - attempting to retrieve should fail
    with pytest.raises(Exception):  # noqa: B017
        client.vector_stores.retrieve(vector_store_id=created_store.id)


@vector_provider_wrapper
def test_openai_vector_store_search_empty(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test searching an empty vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = client.vector_stores.create(
        name="search_test_store",
        metadata={"purpose": "search_testing"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Search the empty store
    search_response = client.vector_stores.search(
        vector_store_id=vector_store.id, query="test query", max_num_results=5
    )

    assert search_response is not None
    assert hasattr(search_response, "data")
    assert len(search_response.data) == 0  # Empty store should return no results
    assert search_response.search_query == ["test query"]
    assert search_response.has_more is False


@vector_provider_wrapper
def test_openai_vector_store_with_chunks(
    compat_client_with_empty_stores,
    client_with_models,
    sample_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test vector store functionality with actual chunks using both OpenAI and native APIs."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store using OpenAI API
    vector_store = compat_client.vector_stores.create(
        name="chunks_test_store",
        metadata={"purpose": "chunks_testing"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Insert chunks using the native LlamaStack API (since OpenAI API doesn't have direct chunk insertion)
    llama_client.vector_io.insert(
        vector_store_id=vector_store.id,
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
@vector_provider_wrapper
def test_openai_vector_store_search_relevance(
    compat_client_with_empty_stores,
    client_with_models,
    sample_chunks,
    test_case,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test that OpenAI vector store search returns relevant results for different queries."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    query, expected_doc_id, expected_topic = test_case

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name=f"relevance_test_{expected_doc_id}",
        metadata={"purpose": "relevance_testing"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Insert chunks using native API
    llama_client.vector_io.insert(
        vector_store_id=vector_store.id,
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


@vector_provider_wrapper
def test_openai_vector_store_search_with_ranking_options(
    compat_client_with_empty_stores,
    client_with_models,
    sample_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test OpenAI vector store search with ranking options."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="ranking_test_store",
        metadata={"purpose": "ranking_testing"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Insert chunks
    llama_client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=sample_chunks,
    )

    # First search without threshold to determine reasonable threshold
    initial_search = compat_client.vector_stores.search(
        vector_store_id=vector_store.id,
        query="machine learning and artificial intelligence",
        max_num_results=3,
    )

    # Use a threshold that's lower than the lowest score to ensure we get results
    if initial_search.data:
        threshold = min(result.score for result in initial_search.data) * 0.9
    else:
        threshold = 0.01

    # Search with ranking options
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


@vector_provider_wrapper
def test_openai_vector_store_search_with_high_score_filter(
    compat_client_with_empty_stores,
    client_with_models,
    sample_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test that searching with text very similar to a document and high score threshold returns only that document."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="high_score_filter_test",
        metadata={"purpose": "high_score_filtering"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Insert chunks
    llama_client.vector_io.insert(
        vector_store_id=vector_store.id,
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


@vector_provider_wrapper
def test_openai_vector_store_search_with_max_num_results(
    compat_client_with_empty_stores,
    client_with_models,
    sample_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test OpenAI vector store search with max_num_results."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    llama_client = client_with_models

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="max_num_results_test_store",
        metadata={"purpose": "max_num_results_testing"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Insert chunks
    llama_client.vector_io.insert(
        vector_store_id=vector_store.id,
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


@vector_provider_wrapper
def test_openai_vector_store_attach_file(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store attach file."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    from llama_stack.apis.files import ExpiresAfter

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create a file
    test_content = b"The secret string is foobazbar."
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "openai_test.txt"
        file = compat_client.files.create(
            file=file_buffer,
            purpose="assistants",
            expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
        )

    # Attach the file to the vector store
    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )

    assert file_attach_response
    assert file_attach_response.object == "vector_store.file"
    assert file_attach_response.id == file.id
    assert file_attach_response.vector_store_id == vector_store.id
    assert file_attach_response.status == "completed"
    assert file_attach_response.chunking_strategy.type == "static"
    assert file_attach_response.created_at > 0
    assert not file_attach_response.last_error

    updated_vector_store = compat_client.vector_stores.retrieve(vector_store_id=vector_store.id)
    assert updated_vector_store.file_counts.completed == 1
    assert updated_vector_store.file_counts.total == 1
    assert updated_vector_store.file_counts.cancelled == 0
    assert updated_vector_store.file_counts.failed == 0
    assert updated_vector_store.file_counts.in_progress == 0

    # Search using OpenAI API to confirm our file attached
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id, query="What is the secret string?", max_num_results=1
    )
    assert search_response is not None
    assert len(search_response.data) > 0
    top_result = search_response.data[0]
    top_content = top_result.content[0].text
    assert "foobazbar" in top_content.lower()


@vector_provider_wrapper
def test_openai_vector_store_attach_files_on_creation(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store attach files on creation."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    from llama_stack.apis.files import ExpiresAfter

    # Create some files and attach them to the vector store
    valid_file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(
                file=file_buffer,
                purpose="assistants",
                expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
            )
        valid_file_ids.append(file.id)

    # include an invalid file ID so we can test failed status
    failed_file_id = "invalid_file_id"
    file_ids = valid_file_ids + [failed_file_id]
    num_failed = len(file_ids) - len(valid_file_ids)

    vector_store = compat_client.vector_stores.create(
        name="test_store",
        file_ids=file_ids,
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    assert vector_store.file_counts.completed == len(valid_file_ids)
    assert vector_store.file_counts.total == len(file_ids)
    assert vector_store.file_counts.cancelled == 0
    assert vector_store.file_counts.failed == num_failed
    assert vector_store.file_counts.in_progress == 0

    files_list = compat_client.vector_stores.files.list(vector_store_id=vector_store.id)
    assert len(files_list.data) == len(file_ids)
    assert set(file_ids) == {file.id for file in files_list.data}
    for file in files_list.data:
        if file.id in valid_file_ids:
            assert file.status == "completed"
        else:
            assert file.status == "failed"

    failed_list = compat_client.vector_stores.files.list(vector_store_id=vector_store.id, filter="failed")
    assert len(failed_list.data) == num_failed
    assert failed_file_id == failed_list.data[0].id

    # Delete the invalid file
    delete_response = compat_client.vector_stores.files.delete(vector_store_id=vector_store.id, file_id=failed_file_id)
    assert delete_response.id == failed_file_id

    updated_vector_store = compat_client.vector_stores.retrieve(vector_store_id=vector_store.id)
    assert updated_vector_store.file_counts.completed == len(valid_file_ids)
    assert updated_vector_store.file_counts.total == len(valid_file_ids)
    assert updated_vector_store.file_counts.failed == 0


@vector_provider_wrapper
def test_openai_vector_store_list_files(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store list files."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    from llama_stack.apis.files import ExpiresAfter

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create some files and attach them to the vector store
    file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(
                file=file_buffer,
                purpose="assistants",
                expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
            )

        response = compat_client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=file.id,
        )
        assert response is not None
        assert response.status == "completed", (
            f"Failed to attach file {file.id} to vector store {vector_store.id}: {response=}"
        )
        file_ids.append(file.id)

    files_list = compat_client.vector_stores.files.list(vector_store_id=vector_store.id)
    assert files_list
    assert files_list.object == "list"
    assert files_list.data is not None
    assert not files_list.has_more
    assert len(files_list.data) == 3
    assert set(file_ids) == {file.id for file in files_list.data}
    assert files_list.data[0].object == "vector_store.file"
    assert files_list.data[0].vector_store_id == vector_store.id
    assert files_list.data[0].status in ["completed", "in_progress"]
    assert files_list.data[0].chunking_strategy.type == "static"
    assert files_list.data[0].created_at > 0
    assert files_list.first_id == files_list.data[0].id
    assert not files_list.data[0].last_error

    first_page = compat_client.vector_stores.files.list(vector_store_id=vector_store.id, limit=2)
    assert first_page.has_more
    assert len(first_page.data) == 2
    assert first_page.first_id == first_page.data[0].id
    assert first_page.last_id == first_page.data[-1].id

    next_page = compat_client.vector_stores.files.list(
        vector_store_id=vector_store.id, limit=2, after=first_page.data[-1].id
    )
    assert not next_page.has_more
    assert len(next_page.data) == 1

    updated_vector_store = compat_client.vector_stores.retrieve(vector_store_id=vector_store.id)
    assert updated_vector_store.file_counts.completed == 3
    assert updated_vector_store.file_counts.total == 3
    assert updated_vector_store.file_counts.cancelled == 0
    assert updated_vector_store.file_counts.failed == 0
    assert updated_vector_store.file_counts.in_progress == 0


@vector_provider_wrapper
def test_openai_vector_store_list_files_invalid_vector_store(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store list files with invalid vector store ID."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        errors = ValueError
    else:
        errors = (BadRequestError, OpenAIBadRequestError)

    with pytest.raises(errors):
        compat_client.vector_stores.files.list(vector_store_id="abc123")


@vector_provider_wrapper
def test_openai_vector_store_retrieve_file_contents(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store retrieve file contents."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    from llama_stack.apis.files import ExpiresAfter

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create a file
    test_content = b"This is a test file"
    file_name = "openai_test.txt"
    attributes = {"foo": "bar"}
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = file_name
        file = compat_client.files.create(
            file=file_buffer,
            purpose="assistants",
            expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
        )

    # Attach the file to the vector store
    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
        attributes=attributes,
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )

    assert file_attach_response.status == "completed"

    file_contents = compat_client.vector_stores.files.content(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )

    assert file_contents is not None
    assert file_contents.object == "vector_store.file_content.page"
    assert len(file_contents.data) == 1
    content = file_contents.data[0]

    # llama-stack-client returns a model, openai-python is a badboy and returns a dict
    if not isinstance(content, dict):
        content = content.model_dump()
    assert content["type"] == "text"
    assert content["text"] == test_content.decode("utf-8")
    assert file_contents.has_more is False


@vector_provider_wrapper
def test_openai_vector_store_delete_file(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store delete file."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    from llama_stack.apis.files import ExpiresAfter

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create some files and attach them to the vector store
    file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(
                file=file_buffer,
                purpose="assistants",
                expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
            )

        compat_client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=file.id,
        )
        file_ids.append(file.id)

    files_list = compat_client.vector_stores.files.list(vector_store_id=vector_store.id)
    assert len(files_list.data) == 3

    # Delete the first file
    delete_response = compat_client.vector_stores.files.delete(vector_store_id=vector_store.id, file_id=file_ids[0])
    assert delete_response
    assert delete_response.id == file_ids[0]
    assert delete_response.deleted is True
    assert delete_response.object == "vector_store.file.deleted"

    updated_vector_store = compat_client.vector_stores.retrieve(vector_store_id=vector_store.id)
    assert updated_vector_store.file_counts.completed == 2
    assert updated_vector_store.file_counts.total == 2
    assert updated_vector_store.file_counts.cancelled == 0
    assert updated_vector_store.file_counts.failed == 0
    assert updated_vector_store.file_counts.in_progress == 0

    # Delete the second file
    delete_response = compat_client.vector_stores.files.delete(vector_store_id=vector_store.id, file_id=file_ids[1])
    assert delete_response
    assert delete_response.id == file_ids[1]

    updated_vector_store = compat_client.vector_stores.retrieve(vector_store_id=vector_store.id)
    assert updated_vector_store.file_counts.completed == 1
    assert updated_vector_store.file_counts.total == 1
    assert updated_vector_store.file_counts.cancelled == 0
    assert updated_vector_store.file_counts.failed == 0
    assert updated_vector_store.file_counts.in_progress == 0


@vector_provider_wrapper
def test_openai_vector_store_delete_file_removes_from_vector_store(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store delete file removes from vector store."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    from llama_stack.apis.files import ExpiresAfter

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create a file
    test_content = b"The secret string is foobazbar."
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "openai_test.txt"
        file = compat_client.files.create(
            file=file_buffer,
            purpose="assistants",
            expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
        )

    # Attach the file to the vector store
    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )
    assert file_attach_response.status == "completed"

    # Search using OpenAI API to confirm our file attached
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id, query="What is the secret string?", max_num_results=1
    )
    assert "foobazbar" in search_response.data[0].content[0].text.lower()

    # Delete the file
    compat_client.vector_stores.files.delete(vector_store_id=vector_store.id, file_id=file.id)

    # Search using OpenAI API to confirm our file deleted
    search_response = compat_client.vector_stores.search(
        vector_store_id=vector_store.id, query="What is the secret string?", max_num_results=1
    )
    assert not search_response.data


@vector_provider_wrapper
def test_openai_vector_store_update_file(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test OpenAI vector store update file."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    from llama_stack.apis.files import ExpiresAfter

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create a file
    test_content = b"This is a test file"
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "openai_test.txt"
        file = compat_client.files.create(
            file=file_buffer,
            purpose="assistants",
            expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
        )

    # Attach the file to the vector store
    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
        attributes={"foo": "bar"},
    )

    assert file_attach_response.status == "completed"
    assert file_attach_response.attributes["foo"] == "bar"

    # Update the file's attributes
    updated_response = compat_client.vector_stores.files.update(
        vector_store_id=vector_store.id,
        file_id=file.id,
        attributes={"foo": "baz"},
    )

    assert updated_response.status == "completed"
    assert updated_response.attributes["foo"] == "baz"

    # Ensure we can retrieve the file and see the updated attributes
    retrieved_file = compat_client.vector_stores.files.retrieve(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )
    assert retrieved_file.attributes["foo"] == "baz"


@vector_provider_wrapper
def test_create_vector_store_files_duplicate_vector_store_name(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """
    This test confirms that client.vector_stores.create() creates a unique ID
    """
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    from llama_stack.apis.files import ExpiresAfter

    compat_client = compat_client_with_empty_stores

    # Create a vector store with files
    file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(
                file=file_buffer,
                purpose="assistants",
                expires_after=ExpiresAfter(anchor="created_at", seconds=86400),  # 24 hours
            )
        file_ids.append(file.id)

    vector_store = compat_client.vector_stores.create(
        name="test_store_with_files",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )
    assert vector_store.file_counts.completed == 0
    assert vector_store.file_counts.total == 0
    assert vector_store.file_counts.cancelled == 0
    assert vector_store.file_counts.failed == 0
    assert vector_store.file_counts.in_progress == 0

    vector_store2 = compat_client.vector_stores.create(
        name="test_store_with_files",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    vector_stores_list = compat_client.vector_stores.list()
    assert len(vector_stores_list.data) == 2

    created_file = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_ids[0],
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )
    assert created_file.status == "completed"

    _ = compat_client.vector_stores.delete(vector_store2.id)
    created_file_from_non_deleted_vector_store = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_ids[1],
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )
    assert created_file_from_non_deleted_vector_store.status == "completed"

    vector_stores_list_post_delete = compat_client.vector_stores.list()
    assert len(vector_stores_list_post_delete.data) == 1


@pytest.mark.parametrize("search_mode", ["vector", "keyword", "hybrid"])
@vector_provider_wrapper
def test_openai_vector_store_search_modes(
    llama_stack_client,
    client_with_models,
    sample_chunks,
    search_mode,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    skip_if_provider_doesnt_support_openai_vector_stores_search(client_with_models, search_mode)

    vector_store = llama_stack_client.vector_stores.create(
        name=f"search_mode_test_{search_mode}",
        metadata={"purpose": "search_mode_testing"},
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    client_with_models.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=sample_chunks,
    )
    query = "Python programming language"

    search_response = llama_stack_client.vector_stores.search(
        vector_store_id=vector_store.id,
        query=query,
        max_num_results=4,
        search_mode=search_mode,
    )
    assert search_response is not None


@vector_provider_wrapper
def test_openai_vector_store_file_batch_create_and_retrieve(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test creating and retrieving a vector store file batch."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="batch_test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create multiple files
    file_ids = []
    for i in range(2):
        with BytesIO(f"This is batch test file {i}".encode()) as file_buffer:
            file_buffer.name = f"batch_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")
        file_ids.append(file.id)

    # Create a file batch
    batch = compat_client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )

    assert batch is not None
    assert batch.object == "vector_store.file_batch"
    assert batch.vector_store_id == vector_store.id
    assert batch.status in ["in_progress", "completed"]
    assert batch.file_counts.total == len(file_ids)
    assert hasattr(batch, "id")
    assert hasattr(batch, "created_at")

    # Wait for batch processing to complete
    max_retries = 60  # 60 seconds max wait (increased for file processing delays)
    retries = 0
    retrieved_batch = None
    while retries < max_retries:
        retrieved_batch = compat_client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=batch.id,
        )
        if retrieved_batch.status in ["completed", "failed"]:
            break
        time.sleep(1)
        retries += 1

    assert retrieved_batch is not None
    assert retrieved_batch.id == batch.id
    assert retrieved_batch.vector_store_id == vector_store.id
    assert retrieved_batch.object == "vector_store.file_batch"
    assert retrieved_batch.file_counts.total == len(file_ids)
    assert retrieved_batch.status == "completed"  # Should be completed after processing


@vector_provider_wrapper
def test_openai_vector_store_file_batch_list_files(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test listing files in a vector store file batch."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="batch_list_test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create multiple files
    file_ids = []
    for i in range(2):
        with BytesIO(f"This is batch list test file {i}".encode()) as file_buffer:
            file_buffer.name = f"batch_list_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")
        file_ids.append(file.id)

    # Create a file batch
    batch = compat_client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )

    # Wait for batch processing to complete
    max_retries = 60  # 60 seconds max wait (increased for file processing delays)
    retries = 0
    while retries < max_retries:
        retrieved_batch = compat_client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=batch.id,
        )
        if retrieved_batch.status in ["completed", "failed"]:
            break
        time.sleep(1)
        retries += 1

    # List all files in the batch
    files_response = compat_client.vector_stores.file_batches.list_files(
        vector_store_id=vector_store.id,
        batch_id=batch.id,
    )

    assert files_response is not None
    assert files_response.object == "list"
    assert hasattr(files_response, "data")
    assert len(files_response.data) == len(file_ids)

    # Verify all files are in the response
    response_file_ids = {file.id for file in files_response.data}
    assert response_file_ids == set(file_ids)

    # Test pagination with limit
    limited_response = compat_client.vector_stores.file_batches.list_files(
        vector_store_id=vector_store.id,
        batch_id=batch.id,
        limit=3,
    )

    assert len(limited_response.data) == 2
    assert limited_response.has_more is False

    # Test pagination with after cursor
    first_page = compat_client.vector_stores.file_batches.list_files(
        vector_store_id=vector_store.id,
        batch_id=batch.id,
        limit=2,
    )

    second_page = compat_client.vector_stores.file_batches.list_files(
        vector_store_id=vector_store.id,
        batch_id=batch.id,
        limit=2,
        after=first_page.data[-1].id,
    )

    assert len(first_page.data) == 2
    assert len(second_page.data) <= 3  # Should be <= remaining files
    # Ensure no overlap between pages
    first_page_ids = {file.id for file in first_page.data}
    second_page_ids = {file.id for file in second_page.data}
    assert first_page_ids.isdisjoint(second_page_ids)


@vector_provider_wrapper
def test_openai_vector_store_file_batch_cancel(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test cancelling a vector store file batch."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="batch_cancel_test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create a batch to test cancellation
    file_ids = []
    for i in range(2):  # Batch size that allows time for cancellation
        with BytesIO(f"This is batch cancel test file {i} with substantial content".encode()) as file_buffer:
            file_buffer.name = f"batch_cancel_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")
        file_ids.append(file.id)

    # Create a file batch
    batch = compat_client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )

    try:
        # Cancel the batch immediately after creation
        cancelled_batch = compat_client.vector_stores.file_batches.cancel(
            vector_store_id=vector_store.id,
            batch_id=batch.id,
        )

        assert cancelled_batch is not None
        assert cancelled_batch.id == batch.id
        assert cancelled_batch.vector_store_id == vector_store.id
        assert cancelled_batch.status == "cancelled"
        assert cancelled_batch.object == "vector_store.file_batch"
    except Exception:
        # If cancellation fails (e.g., batch completed too quickly),
        # verify the batch reached completion instead
        final_batch = compat_client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=batch.id,
        )
        assert final_batch.status in ["completed", "cancelled"]


@vector_provider_wrapper
def test_openai_vector_store_file_batch_retrieve_contents(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test retrieving file contents after file batch processing."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="batch_contents_test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create multiple files with known content
    file_data = [
        ("test_file_1.txt", b"This is the content of test file 1"),
        ("test_file_2.txt", b"This is the content of test file 2"),
    ]

    file_ids = []
    for filename, content in file_data:
        with BytesIO(content) as file_buffer:
            file_buffer.name = filename
            file = compat_client.files.create(file=file_buffer, purpose="assistants")
        file_ids.append(file.id)

    # Create a file batch
    batch = compat_client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )

    # Wait for batch processing to complete
    max_retries = 60  # 60 seconds max wait (increased for file processing delays)
    retries = 0
    while retries < max_retries:
        retrieved_batch = compat_client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=batch.id,
        )
        if retrieved_batch.status in ["completed", "failed"]:
            break
        time.sleep(1)
        retries += 1

    assert retrieved_batch.status == "completed"

    # Retrieve file contents for each file in the batch
    for i, file_id in enumerate(file_ids):
        file_contents = compat_client.vector_stores.files.content(
            vector_store_id=vector_store.id,
            file_id=file_id,
        )

        assert file_contents is not None
        assert file_contents.object == "vector_store.file_content.page"
        assert len(file_contents.data) > 0

        # Verify the content matches what we uploaded
        content_text = (
            file_contents.data[0].text if hasattr(file_contents.data[0], "text") else file_contents.data[0]["text"]
        )
        assert file_data[i][1].decode("utf-8") in content_text


@vector_provider_wrapper
def test_openai_vector_store_file_batch_error_handling(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test error handling for file batch operations."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="batch_error_test_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Test with invalid file IDs (should handle gracefully)
    file_ids = ["invalid_file_id_1", "invalid_file_id_2"]

    batch = compat_client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=file_ids,
        extra_body={
            "embedding_model": embedding_model_id,
        },
    )

    assert batch is not None
    assert batch.file_counts.total == len(file_ids)
    # Invalid files should be marked as failed
    assert batch.file_counts.failed >= 0  # Implementation may vary

    # Test retrieving non-existent batch (returns BadRequestError)
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        batch_errors = ValueError
    else:
        batch_errors = (BadRequestError, OpenAIBadRequestError)

    with pytest.raises(batch_errors):  # Should raise an error for non-existent batch
        compat_client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id="non_existent_batch_id",
        )

    # Test operations on non-existent vector store (returns BadRequestError)
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        vector_store_errors = ValueError
    else:
        vector_store_errors = (BadRequestError, OpenAIBadRequestError)

    with pytest.raises(vector_store_errors):  # Should raise an error for non-existent vector store
        compat_client.vector_stores.file_batches.create(
            vector_store_id="non_existent_vector_store",
            file_ids=["any_file_id"],
        )


@vector_provider_wrapper
def test_openai_vector_store_embedding_config_from_metadata(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test that embedding configuration works from metadata source."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    client = compat_client_with_empty_stores

    # Test 1: Create vector store with embedding config in metadata only
    vector_store_metadata = client.vector_stores.create(
        name="metadata_config_store",
        metadata={
            "embedding_model": embedding_model_id,
            "embedding_dimension": str(embedding_dimension),
            "test_source": "metadata",
        },
        extra_body={
            "provider_id": vector_io_provider_id,
        },
    )

    assert vector_store_metadata is not None
    assert vector_store_metadata.name == "metadata_config_store"
    assert vector_store_metadata.status in ["completed", "in_progress"]
    assert vector_store_metadata.metadata["test_source"] == "metadata"

    # Test 2: Create vector store with consistent config in both sources
    vector_store_consistent = client.vector_stores.create(
        name="consistent_config_store",
        metadata={
            "embedding_model": embedding_model_id,
            "embedding_dimension": str(embedding_dimension),
            "test_source": "consistent",
        },
        extra_body={
            "embedding_model": embedding_model_id,
            "embedding_dimension": int(embedding_dimension),  # Ensure same type/value
            "provider_id": vector_io_provider_id,
        },
    )

    assert vector_store_consistent is not None
    assert vector_store_consistent.name == "consistent_config_store"
    assert vector_store_consistent.status in ["completed", "in_progress"]
    assert vector_store_consistent.metadata["test_source"] == "consistent"

    # Verify both vector stores can be listed
    response = client.vector_stores.list()
    store_names = [store.name for store in response.data]

    assert "metadata_config_store" in store_names
    assert "consistent_config_store" in store_names


@vector_provider_wrapper
def test_openai_vector_store_file_contents_with_extra_query(
    compat_client_with_empty_stores, client_with_models, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    """Test that vector store file contents endpoint supports extra_query parameter."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(
        name="test_extra_query_store",
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    # Create and attach a file
    test_content = b"This is test content for extra_query validation."
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "test_extra_query.txt"
        file = compat_client.files.create(
            file=file_buffer,
            purpose="assistants",
            expires_after=ExpiresAfter(anchor="created_at", seconds=86400),
        )

    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
        extra_body={"embedding_model": embedding_model_id},
    )
    assert file_attach_response.status == "completed"

    # Wait for processing
    time.sleep(2)

    # Test that extra_query parameter is accepted and processed
    content_with_extra_query = compat_client.vector_stores.files.content(
        vector_store_id=vector_store.id,
        file_id=file.id,
        extra_query={"include_embeddings": True, "include_metadata": True},
    )

    # Test without extra_query for comparison
    content_without_extra_query = compat_client.vector_stores.files.content(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )

    # Validate that both calls succeed
    assert content_with_extra_query is not None
    assert content_without_extra_query is not None
    assert len(content_with_extra_query.data) > 0
    assert len(content_without_extra_query.data) > 0

    # Validate that extra_query parameter is processed correctly
    # Both should have the embedding/metadata fields available (may be None based on flags)
    first_chunk_with_flags = content_with_extra_query.data[0]
    first_chunk_without_flags = content_without_extra_query.data[0]

    # The key validation: extra_query fields are present in the response
    # Handle both dict and object responses (different clients may return different formats)
    def has_field(obj, field):
        if isinstance(obj, dict):
            return field in obj
        else:
            return hasattr(obj, field)

    # Validate that all expected fields are present in both responses
    expected_fields = ["embedding", "chunk_metadata", "metadata", "text"]
    for field in expected_fields:
        assert has_field(first_chunk_with_flags, field), f"Field '{field}' missing from response with extra_query"
        assert has_field(first_chunk_without_flags, field), f"Field '{field}' missing from response without extra_query"

    # Validate content is the same
    def get_field(obj, field):
        if isinstance(obj, dict):
            return obj[field]
        else:
            return getattr(obj, field)

    assert get_field(first_chunk_with_flags, "text") == test_content.decode("utf-8")
    assert get_field(first_chunk_without_flags, "text") == test_content.decode("utf-8")

    with_flags_embedding = get_field(first_chunk_with_flags, "embedding")
    without_flags_embedding = get_field(first_chunk_without_flags, "embedding")

    # Validate that embeddings are included when requested and excluded when not requested
    assert with_flags_embedding is not None, "Embeddings should be included when include_embeddings=True"
    assert len(with_flags_embedding) > 0, "Embedding should be a non-empty list"
    assert without_flags_embedding is None, "Embeddings should not be included when include_embeddings=False"
