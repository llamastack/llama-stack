# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.vector_io import (
    Chunk,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    VectorStoreChunkingStrategyAuto,
    VectorStoreFileObject,
)
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.providers.inline.vector_io.sqlite_vec.sqlite_vec import VECTOR_DBS_PREFIX

# This test is a unit test for the inline VectorIO providers. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_vector_io_openai_vector_stores.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


@pytest.fixture(autouse=True)
def mock_resume_file_batches(request):
    """Mock the resume functionality to prevent stale file batches from being processed during tests."""
    with patch(
        "llama_stack.providers.utils.memory.openai_vector_store_mixin.OpenAIVectorStoreMixin._resume_incomplete_batches",
        new_callable=AsyncMock,
    ):
        yield


async def test_initialize_index(vector_index):
    await vector_index.initialize()


async def test_add_chunks_query_vector(vector_index, sample_chunks, sample_embeddings):
    vector_index.delete()
    vector_index.initialize()
    await vector_index.add_chunks(sample_chunks, sample_embeddings)
    resp = await vector_index.query_vector(sample_embeddings[0], k=1, score_threshold=-1)
    assert resp.chunks[0].content == sample_chunks[0].content
    vector_index.delete()


async def test_chunk_id_conflict(vector_index, sample_chunks, embedding_dimension):
    embeddings = np.random.rand(len(sample_chunks), embedding_dimension).astype(np.float32)
    await vector_index.add_chunks(sample_chunks, embeddings)
    resp = await vector_index.query_vector(
        np.random.rand(embedding_dimension).astype(np.float32),
        k=len(sample_chunks),
        score_threshold=-1,
    )

    contents = [chunk.content for chunk in resp.chunks]
    assert len(contents) == len(set(contents))


async def test_initialize_adapter_with_existing_kvstore(vector_io_adapter):
    key = f"{VECTOR_DBS_PREFIX}db1"
    dummy = VectorStore(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await vector_io_adapter.kvstore.set(key=key, value=json.dumps(dummy.model_dump()))

    await vector_io_adapter.initialize()


async def test_persistence_across_adapter_restarts(vector_io_adapter):
    await vector_io_adapter.initialize()
    dummy = VectorStore(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await vector_io_adapter.register_vector_store(dummy)
    await vector_io_adapter.shutdown()

    await vector_io_adapter.initialize()
    assert "foo_db" in vector_io_adapter.cache
    await vector_io_adapter.shutdown()


async def test_vector_store_lazy_loading_from_kvstore(vector_io_adapter):
    """
    Test that vector stores can be lazy-loaded from KV store when not in cache.

    Verifies that clearing the cache doesn't break vector store access - they
    can be loaded on-demand from persistent storage.
    """
    await vector_io_adapter.initialize()

    vector_store_id = f"lazy_load_test_{np.random.randint(1e6)}"
    vector_store = VectorStore(
        identifier=vector_store_id,
        provider_id="test_provider",
        embedding_model="test_model",
        embedding_dimension=128,
    )
    await vector_io_adapter.register_vector_store(vector_store)
    assert vector_store_id in vector_io_adapter.cache

    vector_io_adapter.cache.clear()
    assert vector_store_id not in vector_io_adapter.cache

    loaded_index = await vector_io_adapter._get_and_cache_vector_store_index(vector_store_id)
    assert loaded_index is not None
    assert loaded_index.vector_store.identifier == vector_store_id
    assert vector_store_id in vector_io_adapter.cache

    cached_index = await vector_io_adapter._get_and_cache_vector_store_index(vector_store_id)
    assert cached_index is loaded_index

    await vector_io_adapter.shutdown()


async def test_vector_store_preloading_on_initialization(vector_io_adapter):
    """
    Test that vector stores are preloaded from KV store during initialization.

    Verifies that after restart, all vector stores are automatically loaded into
    cache and immediately accessible without requiring lazy loading.
    """
    await vector_io_adapter.initialize()

    vector_store_ids = [f"preload_test_{i}_{np.random.randint(1e6)}" for i in range(3)]
    for vs_id in vector_store_ids:
        vector_store = VectorStore(
            identifier=vs_id,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=128,
        )
        await vector_io_adapter.register_vector_store(vector_store)

    for vs_id in vector_store_ids:
        assert vs_id in vector_io_adapter.cache

    await vector_io_adapter.shutdown()
    await vector_io_adapter.initialize()

    for vs_id in vector_store_ids:
        assert vs_id in vector_io_adapter.cache

    for vs_id in vector_store_ids:
        loaded_index = await vector_io_adapter._get_and_cache_vector_store_index(vs_id)
        assert loaded_index is not None
        assert loaded_index.vector_store.identifier == vs_id

    await vector_io_adapter.shutdown()


async def test_kvstore_none_raises_runtime_error(vector_io_adapter):
    """
    Test that accessing vector stores with uninitialized kvstore raises RuntimeError.

    Verifies proper RuntimeError is raised instead of assertions when kvstore is None.
    """
    await vector_io_adapter.initialize()

    vector_store_id = f"kvstore_none_test_{np.random.randint(1e6)}"
    vector_store = VectorStore(
        identifier=vector_store_id,
        provider_id="test_provider",
        embedding_model="test_model",
        embedding_dimension=128,
    )
    await vector_io_adapter.register_vector_store(vector_store)

    vector_io_adapter.cache.clear()
    vector_io_adapter.kvstore = None

    with pytest.raises(RuntimeError, match="KVStore not initialized"):
        await vector_io_adapter._get_and_cache_vector_store_index(vector_store_id)


async def test_register_and_unregister_vector_store(vector_io_adapter):
    unique_id = f"foo_db_{np.random.randint(1e6)}"
    dummy = VectorStore(
        identifier=unique_id, provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )

    await vector_io_adapter.register_vector_store(dummy)
    assert dummy.identifier in vector_io_adapter.cache
    await vector_io_adapter.unregister_vector_store(dummy.identifier)
    assert dummy.identifier not in vector_io_adapter.cache


async def test_query_unregistered_raises(vector_io_adapter, vector_provider):
    fake_emb = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError):
        await vector_io_adapter.query_chunks("no_such_db", fake_emb)


async def test_insert_chunks_calls_underlying_index(vector_io_adapter):
    fake_index = AsyncMock()
    vector_io_adapter.cache["db1"] = fake_index

    chunks = ["chunk1", "chunk2"]
    await vector_io_adapter.insert_chunks("db1", chunks)

    fake_index.insert_chunks.assert_awaited_once_with(chunks)


async def test_insert_chunks_missing_db_raises(vector_io_adapter):
    vector_io_adapter._get_and_cache_vector_store_index = AsyncMock(return_value=None)

    with pytest.raises(ValueError):
        await vector_io_adapter.insert_chunks("db_not_exist", [])


async def test_insert_chunks_with_missing_document_id(vector_io_adapter):
    """Ensure no KeyError when document_id is missing or in different places."""
    from llama_stack.apis.vector_io import Chunk, ChunkMetadata

    fake_index = AsyncMock()
    vector_io_adapter.cache["db1"] = fake_index

    # Various document_id scenarios that shouldn't crash
    from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunks = [
        Chunk(
            content="has doc_id in metadata",
            chunk_id=generate_chunk_id("doc-1", "has doc_id in metadata"),
            metadata={"document_id": "doc-1"},
        ),
        Chunk(
            content="no doc_id anywhere",
            chunk_id=generate_chunk_id("unknown", "no doc_id anywhere"),
            metadata={"source": "test"},
        ),
        Chunk(
            content="doc_id in chunk_metadata",
            chunk_id=generate_chunk_id("doc-3", "doc_id in chunk_metadata"),
            chunk_metadata=ChunkMetadata(document_id="doc-3"),
        ),
    ]

    # Should work without KeyError
    await vector_io_adapter.insert_chunks("db1", chunks)
    fake_index.insert_chunks.assert_awaited_once()


async def test_document_id_with_invalid_type_raises_error():
    """Ensure TypeError is raised when document_id is not a string."""
    from llama_stack.apis.vector_io import Chunk

    # Integer document_id should raise TypeError
    from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunk = Chunk(content="test", chunk_id=generate_chunk_id("test", "test"), metadata={"document_id": 12345})
    with pytest.raises(TypeError) as exc_info:
        _ = chunk.document_id
    assert "metadata['document_id'] must be a string" in str(exc_info.value)
    assert "got int" in str(exc_info.value)


async def test_query_chunks_calls_underlying_index_and_returns(vector_io_adapter):
    from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

    expected = QueryChunksResponse(chunks=[Chunk(content="c1", chunk_id=generate_chunk_id("test", "c1"))], scores=[0.1])
    fake_index = AsyncMock(query_chunks=AsyncMock(return_value=expected))
    vector_io_adapter.cache["db1"] = fake_index

    response = await vector_io_adapter.query_chunks("db1", "my_query", {"param": 1})

    fake_index.query_chunks.assert_awaited_once_with("my_query", {"param": 1})
    assert response is expected


async def test_query_chunks_missing_db_raises(vector_io_adapter):
    vector_io_adapter._get_and_cache_vector_store_index = AsyncMock(return_value=None)

    with pytest.raises(ValueError):
        await vector_io_adapter.query_chunks("db_missing", "q", None)


async def test_save_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)

    assert openai_vector_store["id"] in vector_io_adapter.openai_vector_stores
    assert vector_io_adapter.openai_vector_stores[openai_vector_store["id"]] == openai_vector_store


async def test_update_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)
    openai_vector_store["description"] = "Updated description"
    await vector_io_adapter._update_openai_vector_store(store_id, openai_vector_store)
    assert vector_io_adapter.openai_vector_stores[openai_vector_store["id"]] == openai_vector_store


async def test_delete_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)
    await vector_io_adapter._delete_openai_vector_store_from_storage(store_id)
    assert openai_vector_store["id"] not in vector_io_adapter.openai_vector_stores


async def test_load_openai_vector_stores(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)
    loaded_stores = await vector_io_adapter._load_openai_vector_stores()
    assert loaded_stores[store_id] == openai_vector_store


async def test_save_openai_vector_store_file(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    # validating we don't raise an exception
    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)


async def test_update_openai_vector_store_file(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    updated_file_info = file_info.copy()
    updated_file_info["filename"] = "updated_test_file.txt"

    await vector_io_adapter._update_openai_vector_store_file(
        store_id,
        file_id,
        updated_file_info,
    )

    loaded_contents = await vector_io_adapter._load_openai_vector_store_file(store_id, file_id)
    assert loaded_contents == updated_file_info
    assert loaded_contents != file_info


async def test_load_openai_vector_store_file_contents(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    loaded_contents = await vector_io_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert loaded_contents == file_contents


async def test_delete_openai_vector_store_file_from_storage(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)
    await vector_io_adapter._delete_openai_vector_store_file_from_storage(store_id, file_id)

    loaded_file_info = await vector_io_adapter._load_openai_vector_store_file(store_id, file_id)
    assert loaded_file_info == {}
    loaded_contents = await vector_io_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert loaded_contents == []


async def test_create_vector_store_file_batch(vector_io_adapter):
    """Test creating a file batch."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2", "file_3"]

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Mock attach method and batch processing to avoid actual processing
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    vector_io_adapter._process_file_batch_async = AsyncMock()

    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    assert batch.vector_store_id == store_id
    assert batch.status == "in_progress"
    assert batch.file_counts.total == len(file_ids)
    assert batch.file_counts.in_progress == len(file_ids)
    assert batch.id in vector_io_adapter.openai_file_batches


async def test_retrieve_vector_store_file_batch(vector_io_adapter):
    """Test retrieving a file batch."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2"]

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()

    # Create batch first
    created_batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # Retrieve batch
    retrieved_batch = await vector_io_adapter.openai_retrieve_vector_store_file_batch(
        batch_id=created_batch.id,
        vector_store_id=store_id,
    )

    assert retrieved_batch.id == created_batch.id
    assert retrieved_batch.vector_store_id == store_id
    assert retrieved_batch.status == "in_progress"


async def test_cancel_vector_store_file_batch(vector_io_adapter):
    """Test cancelling a file batch."""
    store_id = "vs_1234"
    file_ids = ["file_1"]

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Mock both file attachment and batch processing to prevent automatic completion
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    vector_io_adapter._process_file_batch_async = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # Cancel batch
    cancelled_batch = await vector_io_adapter.openai_cancel_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
    )

    assert cancelled_batch.status == "cancelled"


async def test_list_files_in_vector_store_file_batch(vector_io_adapter):
    """Test listing files in a batch."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2"]

    # Setup vector store with files
    files = {}
    for i, file_id in enumerate(file_ids):
        files[file_id] = VectorStoreFileObject(
            id=file_id,
            object="vector_store.file",
            usage_bytes=1000,
            created_at=int(time.time()) + i,
            vector_store_id=store_id,
            status="completed",
            chunking_strategy=VectorStoreChunkingStrategyAuto(),
        )

    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": files,
        "file_ids": file_ids,
    }

    # Mock file loading
    vector_io_adapter._load_openai_vector_store_file = AsyncMock(
        side_effect=lambda vs_id, f_id: files[f_id].model_dump()
    )
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # List files
    response = await vector_io_adapter.openai_list_files_in_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
    )

    assert len(response.data) == len(file_ids)
    assert response.first_id is not None
    assert response.last_id is not None


async def test_file_batch_validation_errors(vector_io_adapter):
    """Test file batch validation errors."""
    # Test nonexistent vector store
    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.openai_create_vector_store_file_batch(
            vector_store_id="nonexistent",
            params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_1"]),
        )

    # Setup store for remaining tests
    store_id = "vs_test"
    vector_io_adapter.openai_vector_stores[store_id] = {"id": store_id, "files": {}, "file_ids": []}

    # Test nonexistent batch
    with pytest.raises(ValueError, match="File batch .* not found"):
        await vector_io_adapter.openai_retrieve_vector_store_file_batch(
            batch_id="nonexistent_batch",
            vector_store_id=store_id,
        )

    # Test wrong vector store for batch
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_1"])
    )

    # Create wrong_store so it exists but the batch doesn't belong to it
    wrong_store_id = "wrong_store"
    vector_io_adapter.openai_vector_stores[wrong_store_id] = {"id": wrong_store_id, "files": {}, "file_ids": []}

    with pytest.raises(ValueError, match="does not belong to vector store"):
        await vector_io_adapter.openai_retrieve_vector_store_file_batch(
            batch_id=batch.id,
            vector_store_id=wrong_store_id,
        )


async def test_file_batch_pagination(vector_io_adapter):
    """Test file batch pagination."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2", "file_3", "file_4", "file_5"]

    # Setup vector store with multiple files
    files = {}
    for i, file_id in enumerate(file_ids):
        files[file_id] = VectorStoreFileObject(
            id=file_id,
            object="vector_store.file",
            usage_bytes=1000,
            created_at=int(time.time()) + i,
            vector_store_id=store_id,
            status="completed",
            chunking_strategy=VectorStoreChunkingStrategyAuto(),
        )

    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": files,
        "file_ids": file_ids,
    }

    # Mock file loading
    vector_io_adapter._load_openai_vector_store_file = AsyncMock(
        side_effect=lambda vs_id, f_id: files[f_id].model_dump()
    )
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # Test pagination with limit
    response = await vector_io_adapter.openai_list_files_in_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
        limit=3,
    )

    assert len(response.data) == 3
    assert response.has_more is True

    # Test pagination with after cursor
    first_page = await vector_io_adapter.openai_list_files_in_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
        limit=2,
    )

    second_page = await vector_io_adapter.openai_list_files_in_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
        limit=2,
        after=first_page.last_id,
    )

    assert len(first_page.data) == 2
    assert len(second_page.data) == 2
    # Ensure no overlap between pages
    first_page_ids = {file_obj.id for file_obj in first_page.data}
    second_page_ids = {file_obj.id for file_obj in second_page.data}
    assert first_page_ids.isdisjoint(second_page_ids)
    # Verify we got all expected files across both pages (in desc order: file_5, file_4, file_3, file_2, file_1)
    all_returned_ids = first_page_ids | second_page_ids
    assert all_returned_ids == {"file_2", "file_3", "file_4", "file_5"}


async def test_file_batch_status_filtering(vector_io_adapter):
    """Test file batch status filtering."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2", "file_3"]

    # Setup vector store with files having different statuses
    files = {}
    statuses = ["completed", "in_progress", "completed"]
    for i, (file_id, status) in enumerate(zip(file_ids, statuses, strict=False)):
        files[file_id] = VectorStoreFileObject(
            id=file_id,
            object="vector_store.file",
            usage_bytes=1000,
            created_at=int(time.time()) + i,
            vector_store_id=store_id,
            status=status,
            chunking_strategy=VectorStoreChunkingStrategyAuto(),
        )

    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": files,
        "file_ids": file_ids,
    }

    # Mock file loading
    vector_io_adapter._load_openai_vector_store_file = AsyncMock(
        side_effect=lambda vs_id, f_id: files[f_id].model_dump()
    )
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # Test filtering by completed status
    response = await vector_io_adapter.openai_list_files_in_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
        filter="completed",
    )

    assert len(response.data) == 2  # Only 2 completed files
    for file_obj in response.data:
        assert file_obj.status == "completed"

    # Test filtering by in_progress status
    response = await vector_io_adapter.openai_list_files_in_vector_store_file_batch(
        batch_id=batch.id,
        vector_store_id=store_id,
        filter="in_progress",
    )

    assert len(response.data) == 1  # Only 1 in_progress file
    assert response.data[0].status == "in_progress"


async def test_cancel_completed_batch_fails(vector_io_adapter):
    """Test that cancelling completed batch fails."""
    store_id = "vs_1234"
    file_ids = ["file_1"]

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # Manually update status to completed
    batch_info = vector_io_adapter.openai_file_batches[batch.id]
    batch_info["status"] = "completed"

    # Try to cancel - should fail
    with pytest.raises(ValueError, match="Cannot cancel batch .* with status completed"):
        await vector_io_adapter.openai_cancel_vector_store_file_batch(
            batch_id=batch.id,
            vector_store_id=store_id,
        )


async def test_file_batch_persistence_across_restarts(vector_io_adapter):
    """Test that in-progress file batches are persisted and resumed after restart."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2"]

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Mock attach method and batch processing to avoid actual processing
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    vector_io_adapter._process_file_batch_async = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )
    batch_id = batch.id

    # Verify batch is saved to persistent storage
    assert batch_id in vector_io_adapter.openai_file_batches
    saved_batch_key = f"openai_vector_stores_file_batches:v3::{batch_id}"
    saved_batch = await vector_io_adapter.kvstore.get(saved_batch_key)
    assert saved_batch is not None

    # Verify the saved batch data contains all necessary information
    saved_data = json.loads(saved_batch)
    assert saved_data["id"] == batch_id
    assert saved_data["status"] == "in_progress"
    assert saved_data["file_ids"] == file_ids

    # Simulate restart - clear in-memory cache and reload from persistence
    vector_io_adapter.openai_file_batches.clear()

    # Temporarily restore the real initialize_openai_vector_stores method
    from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin

    real_method = OpenAIVectorStoreMixin.initialize_openai_vector_stores
    await real_method(vector_io_adapter)

    # Re-mock the processing method to prevent any resumed batches from processing
    vector_io_adapter._process_file_batch_async = AsyncMock()

    # Verify batch was restored
    assert batch_id in vector_io_adapter.openai_file_batches
    restored_batch = vector_io_adapter.openai_file_batches[batch_id]
    assert restored_batch["status"] == "in_progress"
    assert restored_batch["id"] == batch_id
    assert vector_io_adapter.openai_file_batches[batch_id]["file_ids"] == file_ids


async def test_cancelled_batch_persists_in_storage(vector_io_adapter):
    """Test that cancelled batches persist in storage with updated status."""
    store_id = "vs_1234"
    file_ids = ["file_1", "file_2"]

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Mock attach method and batch processing to avoid actual processing
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    vector_io_adapter._process_file_batch_async = AsyncMock()

    # Create batch
    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )
    batch_id = batch.id

    # Verify batch is initially saved to persistent storage
    saved_batch_key = f"openai_vector_stores_file_batches:v3::{batch_id}"
    saved_batch = await vector_io_adapter.kvstore.get(saved_batch_key)
    assert saved_batch is not None

    # Cancel the batch
    cancelled_batch = await vector_io_adapter.openai_cancel_vector_store_file_batch(
        batch_id=batch_id,
        vector_store_id=store_id,
    )

    # Verify batch status is cancelled
    assert cancelled_batch.status == "cancelled"

    # Verify batch persists in storage with cancelled status
    updated_batch = await vector_io_adapter.kvstore.get(saved_batch_key)
    assert updated_batch is not None
    batch_data = json.loads(updated_batch)
    assert batch_data["status"] == "cancelled"

    # Batch should remain in memory cache (matches vector store pattern)
    assert batch_id in vector_io_adapter.openai_file_batches
    assert vector_io_adapter.openai_file_batches[batch_id]["status"] == "cancelled"


async def test_only_in_progress_batches_resumed(vector_io_adapter):
    """Test that only in-progress batches are resumed for processing, but all batches are persisted."""
    store_id = "vs_1234"

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Mock attach method and batch processing to prevent automatic completion
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    vector_io_adapter._process_file_batch_async = AsyncMock()

    # Create multiple batches
    batch1 = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_1"])
    )
    batch2 = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_2"])
    )

    # Complete one batch (should persist with completed status)
    batch1_info = vector_io_adapter.openai_file_batches[batch1.id]
    batch1_info["status"] = "completed"
    await vector_io_adapter._save_openai_vector_store_file_batch(batch1.id, batch1_info)

    # Cancel the other batch (should persist with cancelled status)
    await vector_io_adapter.openai_cancel_vector_store_file_batch(batch_id=batch2.id, vector_store_id=store_id)

    # Create a third batch that stays in progress
    batch3 = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_3"])
    )

    # Simulate restart - clear memory and reload from persistence
    vector_io_adapter.openai_file_batches.clear()

    # Temporarily restore the real initialize_openai_vector_stores method
    from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin

    real_method = OpenAIVectorStoreMixin.initialize_openai_vector_stores
    await real_method(vector_io_adapter)

    # All batches should be restored from persistence
    assert batch1.id in vector_io_adapter.openai_file_batches  # completed, persisted
    assert batch2.id in vector_io_adapter.openai_file_batches  # cancelled, persisted
    assert batch3.id in vector_io_adapter.openai_file_batches  # in-progress, restored

    # Check their statuses
    assert vector_io_adapter.openai_file_batches[batch1.id]["status"] == "completed"
    assert vector_io_adapter.openai_file_batches[batch2.id]["status"] == "cancelled"
    assert vector_io_adapter.openai_file_batches[batch3.id]["status"] == "in_progress"

    # Resume functionality is mocked, so we're only testing persistence


async def test_cleanup_expired_file_batches(vector_io_adapter):
    """Test that expired file batches are cleaned up properly."""
    store_id = "vs_1234"

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Mock processing to prevent automatic completion
    vector_io_adapter.openai_attach_file_to_vector_store = AsyncMock()
    vector_io_adapter._process_file_batch_async = AsyncMock()

    # Create batches with different ages
    import time

    current_time = int(time.time())

    # Create an old expired batch (10 days old)
    old_batch_info = {
        "id": "batch_old",
        "vector_store_id": store_id,
        "status": "completed",
        "created_at": current_time - (10 * 24 * 60 * 60),  # 10 days ago
        "expires_at": current_time - (3 * 24 * 60 * 60),  # Expired 3 days ago
        "file_ids": ["file_1"],
    }

    # Create a recent valid batch
    new_batch_info = {
        "id": "batch_new",
        "vector_store_id": store_id,
        "status": "completed",
        "created_at": current_time - (1 * 24 * 60 * 60),  # 1 day ago
        "expires_at": current_time + (6 * 24 * 60 * 60),  # Expires in 6 days
        "file_ids": ["file_2"],
    }

    # Store both batches in persistent storage
    await vector_io_adapter._save_openai_vector_store_file_batch("batch_old", old_batch_info)
    await vector_io_adapter._save_openai_vector_store_file_batch("batch_new", new_batch_info)

    # Add to in-memory cache
    vector_io_adapter.openai_file_batches["batch_old"] = old_batch_info
    vector_io_adapter.openai_file_batches["batch_new"] = new_batch_info

    # Verify both batches exist before cleanup
    assert "batch_old" in vector_io_adapter.openai_file_batches
    assert "batch_new" in vector_io_adapter.openai_file_batches

    # Run cleanup
    await vector_io_adapter._cleanup_expired_file_batches()

    # Verify expired batch was removed from memory
    assert "batch_old" not in vector_io_adapter.openai_file_batches
    assert "batch_new" in vector_io_adapter.openai_file_batches

    # Verify expired batch was removed from storage
    old_batch_key = "openai_vector_stores_file_batches:v3::batch_old"
    new_batch_key = "openai_vector_stores_file_batches:v3::batch_new"

    old_stored = await vector_io_adapter.kvstore.get(old_batch_key)
    new_stored = await vector_io_adapter.kvstore.get(new_batch_key)

    assert old_stored is None  # Expired batch should be deleted
    assert new_stored is not None  # Valid batch should remain


async def test_expired_batch_access_error(vector_io_adapter):
    """Test that accessing expired batches returns clear error message."""
    store_id = "vs_1234"

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    # Create an expired batch
    import time

    current_time = int(time.time())

    expired_batch_info = {
        "id": "batch_expired",
        "vector_store_id": store_id,
        "status": "completed",
        "created_at": current_time - (10 * 24 * 60 * 60),  # 10 days ago
        "expires_at": current_time - (3 * 24 * 60 * 60),  # Expired 3 days ago
        "file_ids": ["file_1"],
    }

    # Add to in-memory cache (simulating it was loaded before expiration)
    vector_io_adapter.openai_file_batches["batch_expired"] = expired_batch_info

    # Try to access expired batch
    with pytest.raises(ValueError, match="File batch batch_expired has expired after 7 days from creation"):
        vector_io_adapter._get_and_validate_batch("batch_expired", store_id)


async def test_max_concurrent_files_per_batch(vector_io_adapter):
    """Test that file batch processing respects MAX_CONCURRENT_FILES_PER_BATCH limit."""
    import asyncio

    store_id = "vs_1234"

    # Setup vector store
    vector_io_adapter.openai_vector_stores[store_id] = {
        "id": store_id,
        "name": "Test Store",
        "files": {},
        "file_ids": [],
    }

    active_files = 0

    async def mock_attach_file_with_delay(vector_store_id: str, file_id: str, **kwargs):
        """Mock that tracks concurrency and blocks indefinitely to test concurrency limit."""
        nonlocal active_files
        active_files += 1

        # Block indefinitely to test concurrency limit
        await asyncio.sleep(float("inf"))

    # Replace the attachment method
    vector_io_adapter.openai_attach_file_to_vector_store = mock_attach_file_with_delay

    # Create a batch with more files than the concurrency limit
    file_ids = [f"file_{i}" for i in range(8)]  # 8 files, but limit should be 5

    batch = await vector_io_adapter.openai_create_vector_store_file_batch(
        vector_store_id=store_id, params=OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=file_ids)
    )

    # Give time for the semaphore logic to start processing files
    await asyncio.sleep(0.2)

    # Verify that only MAX_CONCURRENT_FILES_PER_BATCH files are processing concurrently
    # The semaphore in _process_files_with_concurrency should limit this
    from llama_stack.providers.utils.memory.openai_vector_store_mixin import MAX_CONCURRENT_FILES_PER_BATCH

    assert active_files == MAX_CONCURRENT_FILES_PER_BATCH, (
        f"Expected {MAX_CONCURRENT_FILES_PER_BATCH} active files, got {active_files}"
    )

    # Verify batch is in progress
    assert batch.status == "in_progress"
    assert batch.file_counts.total == 8
    assert batch.file_counts.in_progress == 8


async def test_embedding_config_from_metadata(vector_io_adapter):
    """Test that embedding configuration is correctly extracted from metadata."""

    # Mock register_vector_store to avoid actual registration
    vector_io_adapter.register_vector_store = AsyncMock()
    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with embedding config in metadata
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={
            "embedding_model": "test-embedding-model",
            "embedding_dimension": "512",
        },
        model_extra={},
    )

    await vector_io_adapter.openai_create_vector_store(params)

    # Verify VectorStore was registered with correct embedding config from metadata
    vector_io_adapter.register_vector_store.assert_called_once()
    call_args = vector_io_adapter.register_vector_store.call_args[0][0]
    assert call_args.embedding_model == "test-embedding-model"
    assert call_args.embedding_dimension == 512


async def test_embedding_config_from_extra_body(vector_io_adapter):
    """Test that embedding configuration is correctly extracted from extra_body when metadata is empty."""

    # Mock register_vector_store to avoid actual registration
    vector_io_adapter.register_vector_store = AsyncMock()
    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with embedding config in extra_body only (metadata has no embedding_model)
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={},  # Empty metadata to ensure extra_body is used
        **{
            "embedding_model": "extra-body-model",
            "embedding_dimension": 1024,
        },
    )

    await vector_io_adapter.openai_create_vector_store(params)

    # Verify VectorStore was registered with correct embedding config from extra_body
    vector_io_adapter.register_vector_store.assert_called_once()
    call_args = vector_io_adapter.register_vector_store.call_args[0][0]
    assert call_args.embedding_model == "extra-body-model"
    assert call_args.embedding_dimension == 1024


async def test_embedding_config_consistency_check_passes(vector_io_adapter):
    """Test that consistent embedding config in both metadata and extra_body passes validation."""

    # Mock register_vector_store to avoid actual registration
    vector_io_adapter.register_vector_store = AsyncMock()
    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with consistent embedding config in both metadata and extra_body
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={
            "embedding_model": "consistent-model",
            "embedding_dimension": "768",
        },
        **{
            "embedding_model": "consistent-model",
            "embedding_dimension": 768,
        },
    )

    await vector_io_adapter.openai_create_vector_store(params)

    # Should not raise any error and use metadata config
    vector_io_adapter.register_vector_store.assert_called_once()
    call_args = vector_io_adapter.register_vector_store.call_args[0][0]
    assert call_args.embedding_model == "consistent-model"
    assert call_args.embedding_dimension == 768


async def test_embedding_config_defaults_when_missing(vector_io_adapter):
    """Test that embedding dimension defaults to 768 when not provided."""

    # Mock register_vector_store to avoid actual registration
    vector_io_adapter.register_vector_store = AsyncMock()
    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"

    # Test with only embedding model, no dimension (metadata empty to use extra_body)
    params = OpenAICreateVectorStoreRequestWithExtraBody(
        name="test_store",
        metadata={},  # Empty metadata to ensure extra_body is used
        **{
            "embedding_model": "model-without-dimension",
        },
    )

    await vector_io_adapter.openai_create_vector_store(params)

    # Should default to 768 dimensions
    vector_io_adapter.register_vector_store.assert_called_once()
    call_args = vector_io_adapter.register_vector_store.call_args[0][0]
    assert call_args.embedding_model == "model-without-dimension"
    assert call_args.embedding_dimension == 768


async def test_embedding_config_required_model_missing(vector_io_adapter):
    """Test that missing embedding model raises error."""

    # Mock register_vector_store to avoid actual registration
    vector_io_adapter.register_vector_store = AsyncMock()
    # Set provider_id attribute for the adapter
    vector_io_adapter.__provider_id__ = "test_provider"
    # Mock the default model lookup to return None (no default model available)
    vector_io_adapter._get_default_embedding_model_and_dimension = AsyncMock(return_value=None)

    # Test with no embedding model provided
    params = OpenAICreateVectorStoreRequestWithExtraBody(name="test_store", metadata={})

    with pytest.raises(ValueError, match="embedding_model is required"):
        await vector_io_adapter.openai_create_vector_store(params)
