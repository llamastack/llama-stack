# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import random
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from chromadb import PersistentClient
from pymilvus import MilvusClient, connections

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, ChunkMetadata, QueryChunksResponse
from llama_stack.providers.inline.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.faiss.faiss import FaissIndex, FaissVectorIOAdapter
from llama_stack.providers.inline.vector_io.milvus.config import MilvusVectorIOConfig, SqliteKVStoreConfig
from llama_stack.providers.inline.vector_io.qdrant import QdrantVectorIOConfig
from llama_stack.providers.inline.vector_io.sqlite_vec import SQLiteVectorIOConfig
from llama_stack.providers.inline.vector_io.sqlite_vec.sqlite_vec import SQLiteVecIndex, SQLiteVecVectorIOAdapter
from llama_stack.providers.remote.vector_io.chroma.chroma import ChromaIndex, ChromaVectorIOAdapter, maybe_await
from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusIndex, MilvusVectorIOAdapter
from llama_stack.providers.remote.vector_io.pgvector.config import PGVectorVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex, PGVectorVectorIOAdapter
from llama_stack.providers.remote.vector_io.qdrant.qdrant import QdrantVectorIOAdapter
from llama_stack.providers.remote.vector_io.weaviate.config import WeaviateVectorIOConfig
from llama_stack.providers.remote.vector_io.weaviate.weaviate import WeaviateIndex, WeaviateVectorIOAdapter

EMBEDDING_DIMENSION = 384
COLLECTION_PREFIX = "test_collection"
MILVUS_ALIAS = "test_milvus"


@pytest.fixture(params=["milvus", "sqlite_vec", "faiss", "chroma", "pgvector", "weaviate"])
def vector_provider(request):
    return request.param


@pytest.fixture
def vector_db_id() -> str:
    return f"test-vector-db-{random.randint(1, 100)}"


@pytest.fixture(scope="session")
def embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


@pytest.fixture(scope="session")
def sample_chunks():
    """Generates chunks that force multiple batches for a single document to expose ID conflicts."""
    n, k = 10, 3
    sample = [
        Chunk(content=f"Sentence {i} from document {j}", metadata={"document_id": f"document-{j}"})
        for j in range(k)
        for i in range(n)
    ]
    sample.extend(
        [
            Chunk(
                content=f"Sentence {i} from document {j + k}",
                chunk_metadata=ChunkMetadata(
                    document_id=f"document-{j + k}",
                    chunk_id=f"document-{j}-chunk-{i}",
                    source=f"example source-{j + k}-{i}",
                ),
            )
            for j in range(k)
            for i in range(n)
        ]
    )
    return sample


@pytest.fixture(scope="session")
def sample_chunks_with_metadata():
    """Generates chunks that force multiple batches for a single document to expose ID conflicts."""
    n, k = 10, 3
    sample = [
        Chunk(
            content=f"Sentence {i} from document {j}",
            metadata={"document_id": f"document-{j}"},
            chunk_metadata=ChunkMetadata(
                document_id=f"document-{j}",
                chunk_id=f"document-{j}-chunk-{i}",
                source=f"example source-{j}-{i}",
            ),
        )
        for j in range(k)
        for i in range(n)
    ]
    return sample


@pytest.fixture(scope="session")
def sample_embeddings(sample_chunks):
    np.random.seed(42)
    return np.array([np.random.rand(EMBEDDING_DIMENSION).astype(np.float32) for _ in sample_chunks])


@pytest.fixture(scope="session")
def sample_embeddings_with_metadata(sample_chunks_with_metadata):
    np.random.seed(42)
    return np.array([np.random.rand(EMBEDDING_DIMENSION).astype(np.float32) for _ in sample_chunks_with_metadata])


@pytest.fixture(scope="session")
def mock_inference_api(embedding_dimension):
    class MockInferenceAPI:
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [np.random.rand(embedding_dimension).astype(np.float32).tolist() for _ in texts]

    return MockInferenceAPI()


@pytest.fixture
async def unique_kvstore_config(tmp_path_factory):
    # Generate a unique filename for this test
    unique_id = f"test_kv_{np.random.randint(1e6)}"
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"{unique_id}.db")

    return SqliteKVStoreConfig(db_path=db_path)


@pytest.fixture(scope="session")
def sqlite_vec_db_path(tmp_path_factory):
    db_path = str(tmp_path_factory.getbasetemp() / "test_sqlite_vec.db")
    return db_path


@pytest.fixture
async def sqlite_vec_vec_index(embedding_dimension, tmp_path_factory):
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"test_sqlite_vec_{np.random.randint(1e6)}.db")
    bank_id = f"sqlite_vec_bank_{np.random.randint(1e6)}"
    index = SQLiteVecIndex(embedding_dimension, db_path, bank_id)
    await index.initialize()
    index.db_path = db_path
    yield index
    index.delete()


@pytest.fixture
async def sqlite_vec_adapter(sqlite_vec_db_path, mock_inference_api, embedding_dimension):
    config = SQLiteVectorIOConfig(
        db_path=sqlite_vec_db_path,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = SQLiteVecVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    collection_id = f"sqlite_test_collection_{np.random.randint(1e6)}"
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=collection_id,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    adapter.test_collection_id = collection_id
    yield adapter
    await adapter.shutdown()


@pytest.fixture(scope="session")
def milvus_vec_db_path(tmp_path_factory):
    db_path = str(tmp_path_factory.getbasetemp() / "test_milvus.db")
    return db_path


@pytest.fixture
async def milvus_vec_index(milvus_vec_db_path, embedding_dimension):
    client = MilvusClient(milvus_vec_db_path)
    name = f"{COLLECTION_PREFIX}_{np.random.randint(1e6)}"
    connections.connect(alias=MILVUS_ALIAS, uri=milvus_vec_db_path)
    index = MilvusIndex(client, name, consistency_level="Strong")
    index.db_path = milvus_vec_db_path
    yield index


@pytest.fixture
async def milvus_vec_adapter(milvus_vec_db_path, mock_inference_api):
    config = MilvusVectorIOConfig(
        db_path=milvus_vec_db_path,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = MilvusVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=adapter.metadata_collection_name,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=128,
        )
    )
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def faiss_vec_db_path(tmp_path_factory):
    db_path = str(tmp_path_factory.getbasetemp() / "test_faiss.db")
    return db_path


@pytest.fixture
async def faiss_vec_index(embedding_dimension):
    index = FaissIndex(embedding_dimension)
    yield index
    await index.delete()


@pytest.fixture
async def faiss_vec_adapter(unique_kvstore_config, mock_inference_api, embedding_dimension):
    config = FaissVectorIOConfig(
        kvstore=unique_kvstore_config,
    )
    adapter = FaissVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=f"faiss_test_collection_{np.random.randint(1e6)}",
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def chroma_vec_db_path(tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp(f"chroma_{np.random.randint(1e6)}")
    return str(persist_dir)


@pytest.fixture
async def chroma_vec_index(chroma_vec_db_path, embedding_dimension):
    client = PersistentClient(path=chroma_vec_db_path)
    name = f"{COLLECTION_PREFIX}_{np.random.randint(1e6)}"
    collection = await maybe_await(client.get_or_create_collection(name))
    index = ChromaIndex(client=client, collection=collection)
    await index.initialize()
    yield index
    await index.delete()


@pytest.fixture
async def chroma_vec_adapter(chroma_vec_db_path, mock_inference_api, embedding_dimension):
    config = ChromaVectorIOConfig(
        db_path=chroma_vec_db_path,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = ChromaVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=f"chroma_test_collection_{random.randint(1, 1_000_000)}",
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def qdrant_vec_db_path(tmp_path_factory):
    import uuid

    db_path = str(tmp_path_factory.getbasetemp() / f"test_qdrant_{uuid.uuid4()}.db")
    return db_path


@pytest.fixture
async def qdrant_vec_adapter(qdrant_vec_db_path, mock_inference_api, embedding_dimension):
    import uuid

    config = QdrantVectorIOConfig(
        db_path=qdrant_vec_db_path,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = QdrantVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    collection_id = f"qdrant_test_collection_{uuid.uuid4()}"
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=collection_id,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    adapter.test_collection_id = collection_id
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def qdrant_vec_index(qdrant_vec_db_path, embedding_dimension):
    import uuid

    from qdrant_client import AsyncQdrantClient

    from llama_stack.providers.remote.vector_io.qdrant.qdrant import QdrantIndex

    client = AsyncQdrantClient(path=qdrant_vec_db_path)
    collection_name = f"qdrant_test_collection_{uuid.uuid4()}"
    index = QdrantIndex(client, collection_name)
    yield index
    await index.delete()


@pytest.fixture
def mock_psycopg2_connection():
    connection = MagicMock()
    cursor = MagicMock()

    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock()

    connection.cursor.return_value = cursor

    return connection, cursor


@pytest.fixture
async def pgvector_vec_index(embedding_dimension, mock_psycopg2_connection):
    connection, cursor = mock_psycopg2_connection

    vector_db = VectorDB(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=embedding_dimension,
        provider_id="pgvector",
        provider_resource_id="pgvector:test-vector-db",
    )

    with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.execute_values"):
            index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="COSINE")
            index._test_chunks = []
            original_add_chunks = index.add_chunks

            async def mock_add_chunks(chunks, embeddings):
                index._test_chunks = list(chunks)
                await original_add_chunks(chunks, embeddings)

            index.add_chunks = mock_add_chunks

            async def mock_query_vector(embedding, k, score_threshold):
                chunks = index._test_chunks[:k] if hasattr(index, "_test_chunks") else []
                scores = [1.0] * len(chunks)
                return QueryChunksResponse(chunks=chunks, scores=scores)

            index.query_vector = mock_query_vector

    yield index


@pytest.fixture
async def pgvector_vec_adapter(mock_inference_api, embedding_dimension):
    config = PGVectorVectorIOConfig(
        host="localhost",
        port=5432,
        db="test_db",
        user="test_user",
        password="test_password",
        kvstore=SqliteKVStoreConfig(),
    )

    adapter = PGVectorVectorIOAdapter(config, mock_inference_api, None)

    with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.autocommit = True
        mock_connect.return_value = mock_conn

        with patch(
            "llama_stack.providers.remote.vector_io.pgvector.pgvector.check_extension_version"
        ) as mock_check_version:
            mock_check_version.return_value = "0.5.1"

            with patch("llama_stack.providers.utils.kvstore.kvstore_impl") as mock_kvstore_impl:
                mock_kvstore = AsyncMock()
                mock_kvstore_impl.return_value = mock_kvstore

                with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
                    with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.upsert_models"):
                        await adapter.initialize()
                        adapter.conn = mock_conn

                        async def mock_insert_chunks(vector_db_id, chunks, ttl_seconds=None):
                            index = await adapter._get_and_cache_vector_db_index(vector_db_id)
                            if not index:
                                raise ValueError(f"Vector DB {vector_db_id} not found")
                            await index.insert_chunks(chunks)

                        adapter.insert_chunks = mock_insert_chunks

                        async def mock_query_chunks(vector_db_id, query, params=None):
                            index = await adapter._get_and_cache_vector_db_index(vector_db_id)
                            if not index:
                                raise ValueError(f"Vector DB {vector_db_id} not found")
                            return await index.query_chunks(query, params)

                        adapter.query_chunks = mock_query_chunks

                        test_vector_db = VectorDB(
                            identifier=f"pgvector_test_collection_{random.randint(1, 1_000_000)}",
                            provider_id="test_provider",
                            embedding_model="test_model",
                            embedding_dimension=embedding_dimension,
                        )
                        await adapter.register_vector_db(test_vector_db)
                        adapter.test_collection_id = test_vector_db.identifier

                        yield adapter
                        await adapter.shutdown()
def weaviate_vec_db_path():
    return "localhost:8080"


@pytest.fixture
async def weaviate_vec_index(weaviate_vec_db_path, embedding_dimension):
    import uuid

    import weaviate

    # Connect to local Weaviate instance
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
    )

    collection_name = f"{COLLECTION_PREFIX}_{uuid.uuid4()}"
    index = WeaviateIndex(client=client, collection_name=collection_name)

    # Create the collection for this test
    import weaviate.classes as wvc
    from weaviate.collections.classes.config import _CollectionConfig

    from llama_stack.providers.utils.vector_io.vector_utils import sanitize_collection_name

    sanitized_name = sanitize_collection_name(collection_name, weaviate_format=True)
    collection_config = _CollectionConfig(
        name=sanitized_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(
                name="chunk_content",
                data_type=wvc.config.DataType.TEXT,
            ),
        ],
    )
    if not client.collections.exists(sanitized_name):
        client.collections.create_from_config(collection_config)

    yield index
    await index.delete()
    client.close()


@pytest.fixture
async def weaviate_vec_adapter(weaviate_vec_db_path, mock_inference_api, embedding_dimension):
    config = WeaviateVectorIOConfig(
        weaviate_cluster_url=weaviate_vec_db_path,
        weaviate_api_key=None,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = WeaviateVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    collection_id = f"weaviate_test_collection_{random.randint(1, 1_000_000)}"
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=collection_id,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    adapter.test_collection_id = collection_id
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def vector_io_adapter(vector_provider, request):
    vector_provider_dict = {
        "milvus": "milvus_vec_adapter",
        "faiss": "faiss_vec_adapter",
        "sqlite_vec": "sqlite_vec_adapter",
        "chroma": "chroma_vec_adapter",
        "qdrant": "qdrant_vec_adapter",
        "pgvector": "pgvector_vec_adapter",
        "weaviate": "weaviate_vec_adapter",
    }
    return request.getfixturevalue(vector_provider_dict[vector_provider])


@pytest.fixture
def vector_index(vector_provider, request):
    """Returns appropriate vector index based on provider parameter"""
    return request.getfixturevalue(f"{vector_provider}_vec_index")
