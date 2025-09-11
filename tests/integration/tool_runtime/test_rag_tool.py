# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import BadRequestError
from llama_stack_client.types import Document


@pytest.fixture(scope="function")
def client_with_empty_registry(client_with_models):
    def clear_registry():
        vector_dbs = [vector_db.identifier for vector_db in client_with_models.vector_dbs.list()]
        for vector_db_id in vector_dbs:
            client_with_models.vector_dbs.unregister(vector_db_id=vector_db_id)

    clear_registry()

    try:
        client_with_models.toolgroups.register(toolgroup_id="builtin::rag", provider_id="rag-runtime")
    except Exception:
        pass

    yield client_with_models

    clear_registry()


@pytest.fixture(scope="session")
def sample_documents():
    return [
        Document(
            document_id="test-doc-1",
            content="Python is a high-level programming language.",
            metadata={"category": "programming", "difficulty": "beginner"},
        ),
        Document(
            document_id="test-doc-2",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
        Document(
            document_id="test-doc-3",
            content="Data structures are fundamental to computer science.",
            metadata={"category": "computer science", "difficulty": "intermediate"},
        ),
        Document(
            document_id="test-doc-4",
            content="Neural networks are inspired by biological neural networks.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
    ]


def assert_valid_chunk_response(response):
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
    for chunk in response.chunks:
        assert isinstance(chunk.content, str)


def assert_valid_text_response(response):
    assert len(response.content) > 0
    assert all(isinstance(chunk.text, str) for chunk in response.content)


def test_vector_db_insert_inline_and_query(
    client_with_empty_registry, sample_documents, embedding_model_id, embedding_dimension
):
    vector_db_name = "test_vector_db"
    vector_db = client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_name,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )
    vector_db_id = vector_db.identifier

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=sample_documents,
        chunk_size_in_tokens=512,
        vector_db_id=vector_db_id,
    )

    # Query with a direct match
    query1 = "programming language"
    response1 = client_with_empty_registry.vector_io.query(
        vector_db_id=vector_db_id,
        query=query1,
    )
    assert_valid_chunk_response(response1)
    assert any("Python" in chunk.content for chunk in response1.chunks)

    # Query with semantic similarity
    query2 = "AI and brain-inspired computing"
    response2 = client_with_empty_registry.vector_io.query(
        vector_db_id=vector_db_id,
        query=query2,
    )
    assert_valid_chunk_response(response2)
    assert any("neural networks" in chunk.content.lower() for chunk in response2.chunks)

    # Query with limit on number of results (max_chunks=2)
    query3 = "computer"
    response3 = client_with_empty_registry.vector_io.query(
        vector_db_id=vector_db_id,
        query=query3,
        params={"max_chunks": 2},
    )
    assert_valid_chunk_response(response3)
    assert len(response3.chunks) <= 2

    # Query with threshold on similarity score
    query4 = "computer"
    response4 = client_with_empty_registry.vector_io.query(
        vector_db_id=vector_db_id,
        query=query4,
        params={"score_threshold": 0.01},
    )
    assert_valid_chunk_response(response4)
    assert all(score >= 0.01 for score in response4.scores)


def test_vector_db_insert_from_url_and_query(
    client_with_empty_registry, sample_documents, embedding_model_id, embedding_dimension
):
    providers = [p for p in client_with_empty_registry.providers.list() if p.api == "vector_io"]
    assert len(providers) > 0

    vector_db_id = "test_vector_db"

    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    # list to check memory bank is successfully registered
    available_vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    # VectorDB is being migrated to VectorStore, so the ID will be different
    # Just check that at least one vector DB was registered
    assert len(available_vector_dbs) > 0
    # Use the actual registered vector_db_id for subsequent operations
    actual_vector_db_id = available_vector_dbs[0]

    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
    ]
    documents = [
        Document(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=actual_vector_db_id,
        chunk_size_in_tokens=512,
    )

    # Query for the name of method
    response1 = client_with_empty_registry.vector_io.query(
        vector_db_id=actual_vector_db_id,
        query="What's the name of the fine-tunning method used?",
    )
    assert_valid_chunk_response(response1)
    assert any("lora" in chunk.content.lower() for chunk in response1.chunks)

    # Query for the name of model
    response2 = client_with_empty_registry.vector_io.query(
        vector_db_id=actual_vector_db_id,
        query="Which Llama model is mentioned?",
    )
    assert_valid_chunk_response(response2)
    assert any("llama2" in chunk.content.lower() for chunk in response2.chunks)


def test_rag_tool_insert_and_query(client_with_empty_registry, embedding_model_id, embedding_dimension):
    providers = [p for p in client_with_empty_registry.providers.list() if p.api == "vector_io"]
    assert len(providers) > 0

    vector_db_id = "test_vector_db"

    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    available_vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    # VectorDB is being migrated to VectorStore, so the ID will be different
    # Just check that at least one vector DB was registered
    assert len(available_vector_dbs) > 0
    # Use the actual registered vector_db_id for subsequent operations
    actual_vector_db_id = available_vector_dbs[0]

    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
    ]
    documents = [
        Document(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={"author": "llama", "source": url},
        )
        for i, url in enumerate(urls)
    ]

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=actual_vector_db_id,
        chunk_size_in_tokens=512,
    )

    response_with_metadata = client_with_empty_registry.tool_runtime.rag_tool.query(
        vector_db_ids=[actual_vector_db_id],
        content="What is the name of the method used for fine-tuning?",
    )
    assert_valid_text_response(response_with_metadata)
    assert any("metadata:" in chunk.text.lower() for chunk in response_with_metadata.content)

    response_without_metadata = client_with_empty_registry.tool_runtime.rag_tool.query(
        vector_db_ids=[actual_vector_db_id],
        content="What is the name of the method used for fine-tuning?",
        query_config={
            "include_metadata_in_content": True,
            "chunk_template": "Result {index}\nContent: {chunk.content}\n",
        },
    )
    assert_valid_text_response(response_without_metadata)
    assert not any("metadata:" in chunk.text.lower() for chunk in response_without_metadata.content)

    with pytest.raises((ValueError, BadRequestError)):
        client_with_empty_registry.tool_runtime.rag_tool.query(
            vector_db_ids=[actual_vector_db_id],
            content="What is the name of the method used for fine-tuning?",
            query_config={
                "chunk_template": "This should raise a ValueError because it is missing the proper template variables",
            },
        )
