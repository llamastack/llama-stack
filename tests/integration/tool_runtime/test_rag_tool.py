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


def test_rag_tool_openai_apis(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_db_id = "test_openai_vector_db"

    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    available_vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    actual_vector_db_id = available_vector_dbs[0]

    # different document formats that should work with OpenAI APIs
    documents = [
        Document(
            document_id="text-doc",
            content="This is a plain text document about machine learning algorithms.",
            metadata={"type": "text", "category": "AI"},
        ),
        Document(
            document_id="url-doc",
            content="https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/chat.rst",
            mime_type="text/plain",
            metadata={"type": "url", "source": "pytorch"},
        ),
        Document(
            document_id="data-url-doc",
            content="data:text/plain;base64,VGhpcyBpcyBhIGRhdGEgVVJMIGRvY3VtZW50IGFib3V0IGRlZXAgbGVhcm5pbmcu",  # "This is a data URL document about deep learning."
            metadata={"type": "data_url", "encoding": "base64"},
        ),
    ]

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=actual_vector_db_id,
        chunk_size_in_tokens=256,
    )

    files_list = client_with_empty_registry.files.list()
    assert len(files_list.data) >= len(documents), (
        f"Expected at least {len(documents)} files, got {len(files_list.data)}"
    )

    vector_store_files = client_with_empty_registry.vector_io.openai_list_files_in_vector_store(
        vector_store_id=actual_vector_db_id
    )
    assert len(vector_store_files.data) >= len(documents), f"Expected at least {len(documents)} files in vector store"

    response = client_with_empty_registry.tool_runtime.rag_tool.query(
        vector_db_ids=[actual_vector_db_id],
        content="Tell me about machine learning and deep learning",
    )

    assert_valid_text_response(response)
    content_text = " ".join([chunk.text for chunk in response.content]).lower()
    assert "machine learning" in content_text or "deep learning" in content_text


def test_rag_tool_exception_handling(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_db_id = "test_exception_handling"

    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    available_vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    actual_vector_db_id = available_vector_dbs[0]

    documents = [
        Document(
            document_id="valid-doc",
            content="This is a valid document that should be processed successfully.",
            metadata={"status": "valid"},
        ),
        Document(
            document_id="invalid-url-doc",
            content="https://nonexistent-domain-12345.com/invalid.txt",
            metadata={"status": "invalid_url"},
        ),
        Document(
            document_id="another-valid-doc",
            content="This is another valid document for testing resilience.",
            metadata={"status": "valid"},
        ),
    ]

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=actual_vector_db_id,
        chunk_size_in_tokens=256,
    )

    response = client_with_empty_registry.tool_runtime.rag_tool.query(
        vector_db_ids=[actual_vector_db_id],
        content="valid document",
    )

    assert_valid_text_response(response)
    content_text = " ".join([chunk.text for chunk in response.content]).lower()
    assert "valid document" in content_text


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


def test_rag_tool_query_generation(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_db_id = "test_query_generation_db"

    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    available_vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    actual_vector_db_id = available_vector_dbs[0]

    documents = [
        Document(
            document_id="ai-doc",
            content="Artificial intelligence and machine learning are transforming technology.",
            metadata={"category": "AI"},
        ),
        Document(
            document_id="banana-doc",
            content="Don't bring a banana to a knife fight.",
            metadata={"category": "wisdom"},
        ),
    ]

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=actual_vector_db_id,
        chunk_size_in_tokens=256,
    )

    response = client_with_empty_registry.tool_runtime.rag_tool.query(
        vector_db_ids=[actual_vector_db_id],
        content="Tell me about AI",
    )

    assert_valid_text_response(response)
    content_text = " ".join([chunk.text for chunk in response.content]).lower()
    assert "artificial intelligence" in content_text or "machine learning" in content_text


def test_rag_tool_pdf_data_url_handling(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_db_id = "test_pdf_data_url_db"

    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    available_vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    actual_vector_db_id = available_vector_dbs[0]

    sample_pdf = b"%PDF-1.3\n3 0 obj\n<</Type /Page\n/Parent 1 0 R\n/Resources 2 0 R\n/Contents 4 0 R>>\nendobj\n4 0 obj\n<</Filter /FlateDecode /Length 115>>\nstream\nx\x9c\x15\xcc1\x0e\x820\x18@\xe1\x9dS\xbcM]jk$\xd5\xd5(\x83!\x86\xa1\x17\xf8\xa3\xa5`LIh+\xd7W\xc6\xf7\r\xef\xc0\xbd\xd2\xaa\xb6,\xd5\xc5\xb1o\x0c\xa6VZ\xe3znn%\xf3o\xab\xb1\xe7\xa3:Y\xdc\x8bm\xeb\xf3&1\xc8\xd7\xd3\x97\xc82\xe6\x81\x87\xe42\xcb\x87Vb(\x12<\xdd<=}Jc\x0cL\x91\xee\xda$\xb5\xc3\xbd\xd7\xe9\x0f\x8d\x97 $\nendstream\nendobj\n1 0 obj\n<</Type /Pages\n/Kids [3 0 R ]\n/Count 1\n/MediaBox [0 0 595.28 841.89]\n>>\nendobj\n5 0 obj\n<</Type /Font\n/BaseFont /Helvetica\n/Subtype /Type1\n/Encoding /WinAnsiEncoding\n>>\nendobj\n2 0 obj\n<<\n/ProcSet [/PDF /Text /ImageB /ImageC /ImageI]\n/Font <<\n/F1 5 0 R\n>>\n/XObject <<\n>>\n>>\nendobj\n6 0 obj\n<<\n/Producer (PyFPDF 1.7.2 http://pyfpdf.googlecode.com/)\n/Title (This is a sample title.)\n/Author (Llama Stack Developers)\n/CreationDate (D:20250312165548)\n>>\nendobj\n7 0 obj\n<<\n/Type /Catalog\n/Pages 1 0 R\n/OpenAction [3 0 R /FitH null]\n/PageLayout /OneColumn\n>>\nendobj\nxref\n0 8\n0000000000 65535 f \n0000000272 00000 n \n0000000455 00000 n \n0000000009 00000 n \n0000000087 00000 n \n0000000359 00000 n \n0000000559 00000 n \n0000000734 00000 n \ntrailer\n<<\n/Size 8\n/Root 7 0 R\n/Info 6 0 R\n>>\nstartxref\n837\n%%EOF\n"

    import base64

    pdf_base64 = base64.b64encode(sample_pdf).decode("utf-8")
    pdf_data_url = f"data:application/pdf;base64,{pdf_base64}"

    documents = [
        Document(
            document_id="test-pdf-data-url",
            content=pdf_data_url,
            metadata={"type": "pdf", "source": "data_url"},
        ),
    ]

    client_with_empty_registry.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=actual_vector_db_id,
        chunk_size_in_tokens=256,
    )

    files_list = client_with_empty_registry.files.list()
    assert len(files_list.data) >= 1, "PDF should have been uploaded to Files API"

    pdf_file = None
    for file in files_list.data:
        if file.filename and "test-pdf-data-url" in file.filename:
            pdf_file = file
            break

    assert pdf_file is not None, "PDF file should be found in Files API"
    assert pdf_file.bytes == len(sample_pdf), f"File size should match original PDF ({len(sample_pdf)} bytes)"

    file_content = client_with_empty_registry.files.retrieve_content(pdf_file.id)
    assert file_content.startswith(b"%PDF-"), "Retrieved file should be a valid PDF"

    vector_store_files = client_with_empty_registry.vector_io.openai_list_files_in_vector_store(
        vector_store_id=actual_vector_db_id
    )
    assert len(vector_store_files.data) >= 1, "PDF should be attached to vector store"

    response = client_with_empty_registry.tool_runtime.rag_tool.query(
        vector_db_ids=[actual_vector_db_id],
        content="sample title",
    )

    assert_valid_text_response(response)
    content_text = " ".join([chunk.text for chunk in response.content]).lower()
    assert "sample title" in content_text or "title" in content_text
