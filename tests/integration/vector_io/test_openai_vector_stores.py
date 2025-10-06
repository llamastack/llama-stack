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

from llama_stack.apis.vector_io import Chunk
from llama_stack.core.library_client import LlamaStackAsLibraryClient
from llama_stack.log import get_logger

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


def test_openai_create_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test creating a vector store using OpenAI API."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = client.vector_stores.create(
        name="Vs_test_vector_store", metadata={"purpose": "testing", "environment": "integration"}
    )

    assert vector_store is not None
    assert vector_store.name == "Vs_test_vector_store"
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


def test_openai_vector_store_attach_file(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store attach file."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(name="test_store")

    # Create a file
    test_content = b"The secret string is foobazbar."
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "openai_test.txt"
        file = compat_client.files.create(file=file_buffer, purpose="assistants")

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
    assert file_attach_response.chunking_strategy.type == "auto"
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


def test_openai_vector_store_attach_files_on_creation(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store attach files on creation."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create some files and attach them to the vector store
    valid_file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")
        valid_file_ids.append(file.id)

    # include an invalid file ID so we can test failed status
    failed_file_id = "invalid_file_id"
    file_ids = valid_file_ids + [failed_file_id]
    num_failed = len(file_ids) - len(valid_file_ids)

    vector_store = compat_client.vector_stores.create(
        name="test_store",
        file_ids=file_ids,
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


def test_openai_vector_store_list_files(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store list files."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(name="test_store")

    # Create some files and attach them to the vector store
    file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")

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
    assert files_list.data[0].status == "completed"
    assert files_list.data[0].chunking_strategy.type == "auto"
    assert files_list.data[0].created_at > 0
    assert files_list.first_id == files_list.data[0].id
    assert not files_list.data[0].last_error

    first_page = compat_client.vector_stores.files.list(vector_store_id=vector_store.id, limit=2)
    assert first_page.has_more
    assert len(first_page.data) == 2
    assert first_page.first_id == first_page.data[0].id
    assert first_page.last_id != first_page.data[-1].id

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


def test_openai_vector_store_list_files_invalid_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store list files with invalid vector store ID."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        errors = ValueError
    else:
        errors = (BadRequestError, OpenAIBadRequestError)

    with pytest.raises(errors):
        compat_client.vector_stores.files.list(vector_store_id="abc123")


def test_openai_vector_store_retrieve_file_contents(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store retrieve file contents."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(name="test_store")

    # Create a file
    test_content = b"This is a test file"
    file_name = "openai_test.txt"
    attributes = {"foo": "bar"}
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = file_name
        file = compat_client.files.create(file=file_buffer, purpose="assistants")

    # Attach the file to the vector store
    file_attach_response = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
        attributes=attributes,
    )

    assert file_attach_response.status == "completed"

    file_contents = compat_client.vector_stores.files.content(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )

    assert file_contents is not None
    assert len(file_contents.content) == 1
    content = file_contents.content[0]

    # llama-stack-client returns a model, openai-python is a badboy and returns a dict
    if not isinstance(content, dict):
        content = content.model_dump()
    assert content["type"] == "text"
    assert content["text"] == test_content.decode("utf-8")
    assert file_contents.filename == file_name
    assert file_contents.attributes == attributes


def test_openai_vector_store_delete_file(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store delete file."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(name="test_store")

    # Create some files and attach them to the vector store
    file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")

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


def test_openai_vector_store_delete_file_removes_from_vector_store(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store delete file removes from vector store."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(name="test_store")

    # Create a file
    test_content = b"The secret string is foobazbar."
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "openai_test.txt"
        file = compat_client.files.create(file=file_buffer, purpose="assistants")

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


def test_openai_vector_store_update_file(compat_client_with_empty_stores, client_with_models):
    """Test OpenAI vector store update file."""
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store
    vector_store = compat_client.vector_stores.create(name="test_store")

    # Create a file
    test_content = b"This is a test file"
    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "openai_test.txt"
        file = compat_client.files.create(file=file_buffer, purpose="assistants")

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


def test_create_vector_store_files_duplicate_vector_store_name(compat_client_with_empty_stores, client_with_models):
    """
    This test confirms that client.vector_stores.create() creates a unique ID
    """
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)

    compat_client = compat_client_with_empty_stores

    # Create a vector store with files
    file_ids = []
    for i in range(3):
        with BytesIO(f"This is a test file {i}".encode()) as file_buffer:
            file_buffer.name = f"openai_test_{i}.txt"
            file = compat_client.files.create(file=file_buffer, purpose="assistants")
        file_ids.append(file.id)

    vector_store = compat_client.vector_stores.create(
        name="test_store_with_files",
    )
    assert vector_store.file_counts.completed == 0
    assert vector_store.file_counts.total == 0
    assert vector_store.file_counts.cancelled == 0
    assert vector_store.file_counts.failed == 0
    assert vector_store.file_counts.in_progress == 0

    vector_store2 = compat_client.vector_stores.create(
        name="test_store_with_files",
    )

    vector_stores_list = compat_client.vector_stores.list()
    assert len(vector_stores_list.data) == 2

    created_file = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_ids[0],
    )
    assert created_file.status == "completed"

    _ = compat_client.vector_stores.delete(vector_store2.id)
    created_file_from_non_deleted_vector_store = compat_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_ids[1],
    )
    assert created_file_from_non_deleted_vector_store.status == "completed"

    vector_stores_list_post_delete = compat_client.vector_stores.list()
    assert len(vector_stores_list_post_delete.data) == 1


@pytest.mark.parametrize("search_mode", ["vector", "keyword", "hybrid"])
def test_openai_vector_store_search_modes(llama_stack_client, client_with_models, sample_chunks, search_mode):
    skip_if_provider_doesnt_support_openai_vector_stores(client_with_models)
    skip_if_provider_doesnt_support_openai_vector_stores_search(client_with_models, search_mode)

    vector_store = llama_stack_client.vector_stores.create(
        name=f"search_mode_test_{search_mode}",
        metadata={"purpose": "search_mode_testing"},
    )

    client_with_models.vector_io.insert(
        vector_db_id=vector_store.id,
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
