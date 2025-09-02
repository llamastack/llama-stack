# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.apis.inference import Inference
from llama_stack.apis.inference.inference import OpenAIEmbeddingData, OpenAIEmbeddingsResponse, OpenAIEmbeddingUsage
from llama_stack.apis.vector_io import (
    QueryChunksResponse,
    VectorDB,
    VectorDBStore,
)
from llama_stack.providers.inline.vector_io.qdrant.config import (
    QdrantVectorIOConfig as InlineQdrantVectorIOConfig,
)
from llama_stack.providers.remote.vector_io.qdrant.qdrant import (
    QdrantVectorIOAdapter,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig

# This test is a unit test for the QdrantVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_qdrant.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


@pytest.fixture
def qdrant_config(tmp_path) -> InlineQdrantVectorIOConfig:
    kvstore_config = SqliteKVStoreConfig(db_name=os.path.join(tmp_path, "test_kvstore.db"))
    return InlineQdrantVectorIOConfig(path=os.path.join(tmp_path, "qdrant.db"), kvstore=kvstore_config)


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def mock_vector_db(vector_db_id) -> MagicMock:
    mock_vector_db = MagicMock(spec=VectorDB)
    mock_vector_db.embedding_model = "embedding_model"
    mock_vector_db.identifier = vector_db_id
    mock_vector_db.embedding_dimension = 384
    mock_vector_db.model_dump_json.return_value = (
        '{"identifier": "' + vector_db_id + '", "embedding_model": "embedding_model", "embedding_dimension": 384}'
    )
    return mock_vector_db


@pytest.fixture
def mock_vector_db_store(mock_vector_db) -> MagicMock:
    mock_store = MagicMock(spec=VectorDBStore)
    mock_store.get_vector_db = AsyncMock(return_value=mock_vector_db)
    return mock_store


@pytest.fixture
def mock_api_service(sample_embeddings):
    mock_api_service = MagicMock(spec=Inference)
    mock_api_service.openai_embeddings = AsyncMock(
        return_value=OpenAIEmbeddingsResponse(
            model="mock-embedding-model",
            data=[OpenAIEmbeddingData(embedding=sample, index=i) for i, sample in enumerate(sample_embeddings)],
            usage=OpenAIEmbeddingUsage(prompt_tokens=10, total_tokens=10),
        )
    )
    return mock_api_service


@pytest.fixture
async def qdrant_adapter(qdrant_config, mock_vector_db_store, mock_api_service, loop) -> QdrantVectorIOAdapter:
    adapter = QdrantVectorIOAdapter(config=qdrant_config, inference_api=mock_api_service, files_api=None)
    adapter.vector_db_store = mock_vector_db_store
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


__QUERY = "Sample query"


@pytest.mark.parametrize("max_query_chunks, expected_chunks", [(2, 2), (100, 60)])
async def test_qdrant_adapter_returns_expected_chunks(
    qdrant_adapter: QdrantVectorIOAdapter,
    vector_db_id,
    sample_chunks,
    sample_embeddings,
    max_query_chunks,
    expected_chunks,
) -> None:
    assert qdrant_adapter is not None
    await qdrant_adapter.insert_chunks(vector_db_id, sample_chunks)

    index = await qdrant_adapter._get_and_cache_vector_db_index(vector_db_id=vector_db_id)
    assert index is not None

    response = await qdrant_adapter.query_chunks(
        query=__QUERY,
        vector_db_id=vector_db_id,
        params={"max_chunks": max_query_chunks, "mode": "vector"},
    )
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == expected_chunks


# To by-pass attempt to convert a Mock to JSON
def _prepare_for_json(value: Any) -> str:
    return str(value)


@patch("llama_stack.providers.utils.telemetry.trace_protocol._prepare_for_json", new=_prepare_for_json)
async def test_qdrant_register_and_unregister_vector_db(
    qdrant_adapter: QdrantVectorIOAdapter,
    mock_vector_db,
    sample_chunks,
) -> None:
    # Initially, no collections
    vector_db_id = mock_vector_db.identifier
    assert len((await qdrant_adapter.client.get_collections()).collections) == 0

    # Register does not create a collection
    assert not (await qdrant_adapter.client.collection_exists(vector_db_id))
    await qdrant_adapter.register_vector_db(mock_vector_db)
    assert not (await qdrant_adapter.client.collection_exists(vector_db_id))

    # First insert creates the collection
    await qdrant_adapter.insert_chunks(vector_db_id, sample_chunks)
    assert await qdrant_adapter.client.collection_exists(vector_db_id)

    # Unregister deletes the collection
    await qdrant_adapter.unregister_vector_db(vector_db_id)
    assert not (await qdrant_adapter.client.collection_exists(vector_db_id))
    assert len((await qdrant_adapter.client.get_collections()).collections) == 0
