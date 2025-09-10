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
