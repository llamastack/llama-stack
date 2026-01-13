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

from llama_stack.core.storage.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.inline.vector_io.qdrant import QdrantVectorIOConfig as InlineQdrantVectorIOConfig
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from llama_stack.providers.utils.vector_io.vector_utils import load_embedded_chunk_with_backward_compat
from llama_stack_api import (
    EmbeddedChunk,
    Files,
    Inference,
    InterleavedContent,
    QueryChunksResponse,
    VectorIO,
    VectorStore,
    VectorStoreChunkingStrategy,
    VectorStoreFileObject,
    VectorStoreNotFoundError,
    VectorStoresProtocolPrivate,
)

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

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        if not chunks:
            return

        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=len(chunks[0].embedding), distance=models.Distance.COSINE),
            )

        points = []
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            points.append(
                PointStruct(
                    id=convert_id(chunk_id),
                    vector=chunk.embedding,  # Already a list[float]
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
                chunk = load_embedded_chunk_with_backward_compat(point.payload["chunk_content"])
            except Exception:
                log.exception("Failed to parse chunk")
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """
        Performs keyword-based search using Qdrant's MatchText filter.

        Uses Qdrant's query_filter with MatchText to search for chunks containing
        the specified text query string in the chunk content.

        Args:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            QueryChunksResponse with chunks and scores matching the keyword query
        """
        try:
            results = (
                await self.client.query_points(
                    collection_name=self.collection_name,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_content.content", match=models.MatchText(text=query_string)
                            )
                        ]
                    ),
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                )
            ).points
        except Exception as e:
            log.error(f"Error querying keyword search in Qdrant collection {self.collection_name}: {e}")
            raise

        chunks, scores = [], []
        for point in results:
            if not isinstance(point, models.ScoredPoint):
                raise RuntimeError(f"Expected ScoredPoint from Qdrant query, got {type(point).__name__}")
            if point.payload is None:
                raise RuntimeError("Qdrant query returned point with no payload")

            try:
                chunk = load_embedded_chunk_with_backward_compat(point.payload["chunk_content"])
            except Exception:
                chunk_id = point.payload.get(CHUNK_ID_KEY, "unknown") if point.payload else "unknown"
                point_id = getattr(point, "id", "unknown")
                log.exception(
                    f"Failed to parse chunk in collection {self.collection_name}: "
                    f"chunk_id={chunk_id}, point_id={point_id}"
                )
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search combining vector similarity and keyword filtering in a single query.

        Uses Qdrant's native capability to combine a vector query with a query_filter,
        allowing vector similarity search to be filtered by keyword matches in one call.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword filtering
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Not used with this approach, but kept for API compatibility
            reranker_params: Not used with this approach, but kept for API compatibility

        Returns:
            QueryChunksResponse with filtered vector search results
        """
        try:
            results = (
                await self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding.tolist(),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_content.content", match=models.MatchText(text=query_string)
                            )
                        ]
                    ),
                    limit=k,
                    with_payload=True,
                    score_threshold=score_threshold,
                )
            ).points
        except Exception as e:
            log.error(f"Error querying hybrid search in Qdrant collection {self.collection_name}: {e}")
            raise

        chunks, scores = [], []
        for point in results:
            if not isinstance(point, models.ScoredPoint):
                raise RuntimeError(f"Expected ScoredPoint from Qdrant query, got {type(point).__name__}")
            if point.payload is None:
                raise RuntimeError("Qdrant query returned point with no payload")

            try:
                chunk = load_embedded_chunk_with_backward_compat(point.payload["chunk_content"])
            except Exception:
                chunk_id = point.payload.get(CHUNK_ID_KEY, "unknown") if point.payload else "unknown"
                point_id = getattr(point, "id", "unknown")
                log.exception(
                    f"Failed to parse chunk in collection {self.collection_name}: "
                    f"chunk_id={chunk_id}, point_id={point_id}"
                )
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        await self.client.delete_collection(collection_name=self.collection_name)


class QdrantVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: RemoteQdrantVectorIOConfig | InlineQdrantVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
    ) -> None:
        super().__init__(inference_api=inference_api, files_api=files_api, kvstore=None)
        self.config = config
        self.client: AsyncQdrantClient = None
        self.cache = {}
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

    async def insert_chunks(
        self, vector_store_id: str, chunks: list[EmbeddedChunk], ttl_seconds: int | None = None
    ) -> None:
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
