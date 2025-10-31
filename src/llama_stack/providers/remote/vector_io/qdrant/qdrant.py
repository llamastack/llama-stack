# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import hashlib
import uuid
from typing import Any

from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
    VectorStoreChunkingStrategy,
    VectorStoreFileObject,
)
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import VectorStoresProtocolPrivate
from llama_stack.providers.inline.vector_io.qdrant import QdrantVectorIOConfig as InlineQdrantVectorIOConfig
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex

from .config import QdrantVectorIOConfig as RemoteQdrantVectorIOConfig

log = get_logger(name=__name__, category="vector_io::qdrant")
CHUNK_ID_KEY = "_chunk_id"

# KV store prefixes for vector databases
VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:qdrant:{VERSION}::"


def convert_id(_id: str) -> str:
    """
    Converts any string into a UUID string based on a seed.

    Qdrant accepts UUID strings and unsigned integers as point ID.
    We use a SHA-256 hash to convert each string into a UUID string deterministically.
    This allows us to overwrite the same point with the original ID.
    """
    hash_input = f"qdrant_id:{_id}".encode()
    sha256_hash = hashlib.sha256(hash_input).hexdigest()
    # Use the first 32 characters to create a valid UUID
    return str(uuid.UUID(sha256_hash[:32]))


class QdrantIndex(EmbeddingIndex):
    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    async def initialize(self) -> None:
        # Qdrant collections are created on-demand in add_chunks
        # If the collection does not exist, it will be created in add_chunks.
        pass

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE),
            )

        points = []
        for _i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = chunk.chunk_id
            points.append(
                PointStruct(
                    id=convert_id(chunk_id),
                    vector=embedding,
                    payload={"chunk_content": chunk.model_dump()} | {CHUNK_ID_KEY: chunk_id},
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the Qdrant collection."""
        chunk_ids = [convert_id(c.chunk_id) for c in chunks_for_deletion]
        try:
            await self.client.delete(
                collection_name=self.collection_name, points_selector=models.PointIdsList(points=chunk_ids)
            )
        except Exception as e:
            log.error(f"Error deleting chunks from Qdrant collection {self.collection_name}: {e}")
            raise

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        results = (
            await self.client.query_points(
                collection_name=self.collection_name,
                query=embedding.tolist(),
                limit=k,
                with_payload=True,
                score_threshold=score_threshold,
            )
        ).points

        chunks, scores = [], []
        for point in results:
            assert isinstance(point, models.ScoredPoint)
            assert point.payload is not None

            try:
                chunk = Chunk(**point.payload["chunk_content"])
            except Exception:
                log.exception("Failed to parse chunk")
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in Qdrant")

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in Qdrant")

    async def delete(self):
        await self.client.delete_collection(collection_name=self.collection_name)


class QdrantVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: RemoteQdrantVectorIOConfig | InlineQdrantVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
    ) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        self.config = config
        self.client: AsyncQdrantClient = None
        self.cache = {}
        self.inference_api = inference_api
        self.vector_store_table = None
        self._qdrant_lock = asyncio.Lock()

    async def initialize(self) -> None:
        client_config = self.config.model_dump(exclude_none=True, exclude={"persistence"})
        self.client = AsyncQdrantClient(**client_config)
        self.kvstore = await kvstore_impl(self.config.persistence)

        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store, QdrantIndex(self.client, vector_store.identifier), self.inference_api
            )
            self.cache[vector_store.identifier] = index
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        await self.client.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=QdrantIndex(self.client, vector_store.identifier),
            inference_api=self.inference_api,
        )

        self.cache[vector_store.identifier] = index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        # Try to load from kvstore
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=QdrantIndex(client=self.client, collection_name=vector_store.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def insert_chunks(self, vector_store_id: str, chunks: list[Chunk], ttl_seconds: int | None = None) -> None:
        index = await self._get_and_cache_vector_store_index(vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(vector_store_id)

        await index.insert_chunks(chunks)

    async def query_chunks(
        self, vector_store_id: str, query: InterleavedContent, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(vector_store_id)

        return await index.query_chunks(query, params)

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> VectorStoreFileObject:
        # Qdrant doesn't allow multiple clients to access the same storage path simultaneously.
        async with self._qdrant_lock:
            return await super().openai_attach_file_to_vector_store(
                vector_store_id, file_id, attributes, chunking_strategy
            )

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from a Qdrant vector store."""
        index = await self._get_and_cache_vector_store_index(store_id)
        if not index:
            raise ValueError(f"Vector DB {store_id} not found")

        await index.index.delete_chunks(chunks_for_deletion)
