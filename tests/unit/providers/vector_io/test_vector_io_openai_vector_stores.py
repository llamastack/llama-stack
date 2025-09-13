# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.remote.vector_io.milvus.milvus import VECTOR_DBS_PREFIX

# This test is a unit test for the inline VectoerIO providers. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_vector_io_openai_vector_stores.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


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
    dummy = VectorDB(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await vector_io_adapter.kvstore.set(key=key, value=json.dumps(dummy.model_dump()))

    await vector_io_adapter.initialize()


async def test_persistence_across_adapter_restarts(vector_io_adapter):
    await vector_io_adapter.initialize()
    dummy = VectorDB(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await vector_io_adapter.register_vector_db(dummy)
    await vector_io_adapter.shutdown()

    await vector_io_adapter.initialize()
    assert "foo_db" in vector_io_adapter.cache
    await vector_io_adapter.shutdown()


async def test_register_and_unregister_vector_db(vector_io_adapter):
    unique_id = f"foo_db_{np.random.randint(1e6)}"
    dummy = VectorDB(
        identifier=unique_id, provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )

    await vector_io_adapter.register_vector_db(dummy)
    assert dummy.identifier in vector_io_adapter.cache
    await vector_io_adapter.unregister_vector_db(dummy.identifier)
    assert dummy.identifier not in vector_io_adapter.cache


async def test_query_unregistered_raises(vector_io_adapter, vector_provider):
    fake_emb = np.zeros(8, dtype=np.float32)
    if vector_provider == "chroma":
        with pytest.raises(AttributeError):
            await vector_io_adapter.query_chunks("no_such_db", fake_emb)
    else:
        with pytest.raises(ValueError):
            await vector_io_adapter.query_chunks("no_such_db", fake_emb)


async def test_insert_chunks_calls_underlying_index(vector_io_adapter):
    fake_index = AsyncMock()
    vector_io_adapter.cache["db1"] = fake_index

    chunks = ["chunk1", "chunk2"]
    await vector_io_adapter.insert_chunks("db1", chunks)

    fake_index.insert_chunks.assert_awaited_once_with(chunks)


async def test_insert_chunks_missing_db_raises(vector_io_adapter):
    vector_io_adapter._get_and_cache_vector_db_index = AsyncMock(return_value=None)

    with pytest.raises(ValueError):
        await vector_io_adapter.insert_chunks("db_not_exist", [])


async def test_query_chunks_calls_underlying_index_and_returns(vector_io_adapter):
    expected = QueryChunksResponse(chunks=[Chunk(content="c1")], scores=[0.1])
    fake_index = AsyncMock(query_chunks=AsyncMock(return_value=expected))
    vector_io_adapter.cache["db1"] = fake_index

    response = await vector_io_adapter.query_chunks("db1", "my_query", {"param": 1})

    fake_index.query_chunks.assert_awaited_once_with("my_query", {"param": 1})
    assert response is expected


async def test_query_chunks_missing_db_raises(vector_io_adapter):
    vector_io_adapter._get_and_cache_vector_db_index = AsyncMock(return_value=None)

    with pytest.raises(ValueError):
        await vector_io_adapter.query_chunks("db_missing", "q", None)


async def test_save_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_db_id": "test_db",
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
        "vector_db_id": "test_db",
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
        "vector_db_id": "test_db",
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
        "vector_db_id": "test_db",
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


async def test_openai_retrieve_vector_store_chunk(vector_io_adapter):
    """Test retrieving a specific chunk from a vector store file."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id = "chunk_001"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {
            "content": {"type": "text", "text": "First chunk content"},
            "stored_chunk_id": chunk_id,
            "metadata": {"file_id": file_id, "position": 0},
            "chunk_metadata": {"chunk_id": chunk_id},
        },
        {
            "content": {"type": "text", "text": "Second chunk content"},
            "stored_chunk_id": "chunk_002",
            "metadata": {"file_id": file_id, "position": 1},
            "chunk_metadata": {"chunk_id": "chunk_002"},
        },
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    chunk_object = await vector_io_adapter.openai_retrieve_vector_store_chunk(
        vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
    )

    assert chunk_object.id == chunk_id
    assert chunk_object.vector_store_id == store_id
    assert chunk_object.file_id == file_id
    assert chunk_object.object == "vector_store.file.chunk"
    assert chunk_object.content.type == "text"
    assert chunk_object.content.text == "First chunk content"
    assert chunk_object.metadata["file_id"] == file_id
    assert chunk_object.metadata["position"] == 0


async def test_openai_retrieve_vector_store_chunk_not_found(vector_io_adapter):
    """Test retrieving a non-existent chunk raises appropriate error."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id = "nonexistent_chunk"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {"id": file_id, "created_at": int(time.time())}
    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, [])

    with pytest.raises(ValueError, match="Chunk nonexistent_chunk not found"):
        await vector_io_adapter.openai_retrieve_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
        )


async def test_openai_update_vector_store_chunk_metadata_only(vector_io_adapter):
    """Test updating only the metadata of a chunk."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id = "chunk_001"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    original_content = "Original chunk content"
    file_contents = [
        {
            "content": {"type": "text", "text": original_content},
            "stored_chunk_id": chunk_id,
            "metadata": {"file_id": file_id, "version": 1},
            "chunk_metadata": {"chunk_id": chunk_id},
        }
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    vector_io_adapter.delete_chunks = AsyncMock()
    vector_io_adapter.insert_chunks = AsyncMock()

    new_metadata = {"file_id": file_id, "version": 2, "updated": True}
    updated_chunk = await vector_io_adapter.openai_update_vector_store_chunk(
        vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id, metadata=new_metadata
    )

    vector_io_adapter.delete_chunks.assert_not_called()
    vector_io_adapter.insert_chunks.assert_not_called()

    assert updated_chunk.id == chunk_id
    assert updated_chunk.metadata["version"] == 2
    assert updated_chunk.metadata["updated"] is True
    assert updated_chunk.content.text == original_content


async def test_openai_update_vector_store_chunk_content(vector_io_adapter):
    """Test updating the content of a chunk."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id = "chunk_001"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {
            "content": {"type": "text", "text": "Original chunk content"},
            "stored_chunk_id": chunk_id,
            "metadata": {"file_id": file_id},
            "chunk_metadata": {"chunk_id": chunk_id},
        }
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    vector_io_adapter.delete_chunks = AsyncMock()
    vector_io_adapter.insert_chunks = AsyncMock()

    new_content = {"type": "text", "text": "Updated chunk content"}
    updated_chunk = await vector_io_adapter.openai_update_vector_store_chunk(
        vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id, content=new_content
    )

    vector_io_adapter.delete_chunks.assert_awaited_once_with(store_id, [chunk_id])
    vector_io_adapter.insert_chunks.assert_awaited_once()

    assert updated_chunk.id == chunk_id
    assert updated_chunk.content.text == "Updated chunk content"


async def test_openai_update_vector_store_chunk_both_content_and_metadata(vector_io_adapter):
    """Test updating both content and metadata of a chunk."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id = "chunk_001"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {
            "content": {"type": "text", "text": "Original chunk content"},
            "stored_chunk_id": chunk_id,
            "metadata": {"file_id": file_id, "version": 1},
            "chunk_metadata": {"chunk_id": chunk_id},
        }
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    vector_io_adapter.delete_chunks = AsyncMock()
    vector_io_adapter.insert_chunks = AsyncMock()

    new_content = {"type": "text", "text": "Updated chunk content"}
    new_metadata = {"file_id": file_id, "version": 2, "updated": True}
    updated_chunk = await vector_io_adapter.openai_update_vector_store_chunk(
        vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id, content=new_content, metadata=new_metadata
    )

    vector_io_adapter.delete_chunks.assert_awaited_once_with(store_id, [chunk_id])
    vector_io_adapter.insert_chunks.assert_awaited_once()

    assert updated_chunk.id == chunk_id
    assert updated_chunk.content.text == "Updated chunk content"
    assert updated_chunk.metadata["version"] == 2
    assert updated_chunk.metadata["updated"] is True


async def test_openai_delete_vector_store_chunk(vector_io_adapter):
    """Test deleting a specific chunk from a vector store file."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id_to_delete = "chunk_001"
    chunk_id_to_keep = "chunk_002"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {
            "content": {"type": "text", "text": "First chunk content"},
            "stored_chunk_id": chunk_id_to_delete,
            "metadata": {"file_id": file_id, "position": 0},
            "chunk_metadata": {"chunk_id": chunk_id_to_delete},
        },
        {
            "content": {"type": "text", "text": "Second chunk content"},
            "stored_chunk_id": chunk_id_to_keep,
            "metadata": {"file_id": file_id, "position": 1},
            "chunk_metadata": {"chunk_id": chunk_id_to_keep},
        },
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    vector_io_adapter.delete_chunks = AsyncMock()

    delete_response = await vector_io_adapter.openai_delete_vector_store_chunk(
        vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id_to_delete
    )

    vector_io_adapter.delete_chunks.assert_awaited_once_with(store_id, [chunk_id_to_delete])

    assert delete_response.id == chunk_id_to_delete
    assert delete_response.object == "vector_store.file.chunk.deleted"
    assert delete_response.deleted is True

    remaining_contents = await vector_io_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert len(remaining_contents) == 1
    assert remaining_contents[0]["stored_chunk_id"] == chunk_id_to_keep


async def test_openai_delete_vector_store_chunk_not_found(vector_io_adapter):
    """Test deleting a non-existent chunk raises appropriate error."""
    store_id = "vs_1234"
    file_id = "file_1234"
    chunk_id = "nonexistent_chunk"

    store_info = {
        "id": store_id,
        "file_ids": [file_id],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    file_info = {"id": file_id, "created_at": int(time.time())}
    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, [])

    with pytest.raises(ValueError, match="Chunk nonexistent_chunk not found"):
        await vector_io_adapter.openai_delete_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
        )


async def test_chunk_operations_with_nonexistent_vector_store(vector_io_adapter):
    """Test that chunk operations raise errors for non-existent vector stores."""

    store_id = "nonexistent_store"
    file_id = "file_1234"
    chunk_id = "chunk_001"

    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.openai_retrieve_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
        )

    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.openai_update_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id, metadata={"test": "value"}
        )

    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.openai_delete_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
        )


async def test_chunk_operations_with_nonexistent_file(vector_io_adapter):
    """Test that chunk operations raise errors for non-existent files."""
    store_id = "vs_1234"
    file_id = "nonexistent_file"
    chunk_id = "chunk_001"

    store_info = {
        "id": store_id,
        "file_ids": [],
        "created_at": int(time.time()),
    }
    vector_io_adapter.openai_vector_stores[store_id] = store_info

    with pytest.raises(ValueError, match=f"File {file_id} not found in vector store"):
        await vector_io_adapter.openai_retrieve_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
        )

    with pytest.raises(ValueError, match=f"File {file_id} not found in vector store"):
        await vector_io_adapter.openai_update_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id, metadata={"test": "value"}
        )

    with pytest.raises(ValueError, match=f"File {file_id} not found in vector store"):
        await vector_io_adapter.openai_delete_vector_store_chunk(
            vector_store_id=store_id, file_id=file_id, chunk_id=chunk_id
        )

    with pytest.raises(ValueError, match=f"File {file_id} not found in vector store"):
        await vector_io_adapter.openai_list_vector_store_chunks(vector_store_id=store_id, file_id=file_id)


async def test_openai_list_vector_store_chunks(vector_io_adapter):
    """Test listing chunks in a vector store file."""
    store_id = "test_store_123"
    await vector_io_adapter.openai_create_vector_store(
        provider_vector_db_id=store_id,
        name="Test Store",
        embedding_model="test_model",
        embedding_dimension=512,
        provider_id="test_provider",
    )

    test_content = "This is test content for chunk listing."
    file_id = "test_file_456"
    test_metadata = {"source": "test_file", "chunk_number": "1", "file_id": file_id}
    test_embedding = [0.1] * 512

    chunk1 = Chunk(
        content=test_content + " First chunk.",
        metadata={**test_metadata, "chunk_id": "1"},
        embedding=test_embedding,
        chunk_id="chunk_1",
    )
    chunk2 = Chunk(
        content=test_content + " Second chunk.",
        metadata={**test_metadata, "chunk_id": "2"},
        embedding=[0.2] * 512,
        chunk_id="chunk_2",
    )
    chunk3 = Chunk(
        content=test_content + " Third chunk.",
        metadata={**test_metadata, "chunk_id": "3"},
        embedding=[0.3] * 512,
        chunk_id="chunk_3",
    )

    await vector_io_adapter.insert_chunks(store_id, [chunk1, chunk2, chunk3])
    file_info = {
        "id": file_id,
        "object": "vector_store.file",
        "created_at": int(time.time()),
        "vector_store_id": store_id,
        "status": "completed",
        "usage_bytes": 1024,
        "chunking_strategy": {"type": "static", "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400}},
        "filename": "test_file.txt",
    }

    dict_chunks = [chunk1.model_dump(), chunk2.model_dump(), chunk3.model_dump()]
    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, dict_chunks)

    vector_io_adapter.openai_vector_stores[store_id]["file_ids"].append(file_id)

    response = await vector_io_adapter.openai_list_vector_store_chunks(vector_store_id=store_id, file_id=file_id)

    assert response.object == "list"
    assert len(response.data) == 3
    assert response.has_more is False
    assert response.first_id is not None
    assert response.last_id is not None

    chunk_ids = [chunk.id for chunk in response.data]
    expected_chunk_ids = {"chunk_1", "chunk_2", "chunk_3", "1", "2", "3"}  # Accept either format
    for chunk_id in chunk_ids:
        assert chunk_id in expected_chunk_ids, f"Unexpected chunk_id: {chunk_id}"

    for chunk in response.data:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 512
        assert chunk.vector_store_id == store_id
        assert chunk.file_id == file_id

    limited_response = await vector_io_adapter.openai_list_vector_store_chunks(
        vector_store_id=store_id, file_id=file_id, limit=2
    )

    assert len(limited_response.data) == 2
    assert limited_response.has_more is True

    desc_response = await vector_io_adapter.openai_list_vector_store_chunks(
        vector_store_id=store_id, file_id=file_id, order="desc"
    )

    assert len(desc_response.data) == 3

    asc_response = await vector_io_adapter.openai_list_vector_store_chunks(
        vector_store_id=store_id, file_id=file_id, order="asc"
    )

    assert len(asc_response.data) == 3

    first_chunk_id = response.data[0].id
    after_response = await vector_io_adapter.openai_list_vector_store_chunks(
        vector_store_id=store_id, file_id=file_id, after=first_chunk_id
    )

    assert len(after_response.data) <= 2
    after_chunk_ids = [chunk.id for chunk in after_response.data]
    assert first_chunk_id not in after_chunk_ids


async def test_openai_list_vector_store_chunks_empty_file(vector_io_adapter):
    """Test listing chunks in an empty file."""
    store_id = "test_store_empty"
    await vector_io_adapter.openai_create_vector_store(
        provider_vector_db_id=store_id,
        name="Test Store",
        embedding_model="test_model",
        embedding_dimension=512,
        provider_id="test_provider",
    )

    file_id = "empty_file"
    file_info = {
        "id": file_id,
        "object": "vector_store.file",
        "created_at": int(time.time()),
        "vector_store_id": store_id,
        "status": "completed",
        "usage_bytes": 0,
        "chunking_strategy": {"type": "static", "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400}},
        "filename": "empty_file.txt",
    }

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, [])

    vector_io_adapter.openai_vector_stores[store_id]["file_ids"].append(file_id)

    response = await vector_io_adapter.openai_list_vector_store_chunks(vector_store_id=store_id, file_id=file_id)

    assert response.object == "list"
    assert len(response.data) == 0
    assert response.has_more is False
    assert response.first_id is None
    assert response.last_id is None


async def test_openai_list_vector_store_chunks_nonexistent_resources(vector_io_adapter):
    with pytest.raises(VectorStoreNotFoundError):
        await vector_io_adapter.openai_list_vector_store_chunks(vector_store_id="nonexistent_store", file_id="any_file")

    store_id = "test_store_list"
    await vector_io_adapter.openai_create_vector_store(
        provider_vector_db_id=store_id,
        name="Test Store",
        embedding_model="test_model",
        embedding_dimension=512,
        provider_id="test_provider",
    )

    with pytest.raises(ValueError, match="File nonexistent_file not found in vector store"):
        await vector_io_adapter.openai_list_vector_store_chunks(vector_store_id=store_id, file_id="nonexistent_file")
