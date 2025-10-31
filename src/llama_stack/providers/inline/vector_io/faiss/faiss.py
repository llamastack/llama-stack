# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import HealthResponse, HealthStatus, VectorStoresProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex

from .config import FaissVectorIOConfig

logger = get_logger(name=__name__, category="vector_io")

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:{VERSION}::"
FAISS_INDEX_PREFIX = f"faiss_index:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:{VERSION}::"


class FaissIndex(EmbeddingIndex):
    def __init__(self, dimension: int, kvstore: KVStore | None = None, bank_id: str | None = None):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunk_by_index: dict[int, Chunk] = {}
        self.kvstore = kvstore
        self.bank_id = bank_id

        # A list of chunk id's in the same order as they are in the index,
        # must be updated when chunks are added or removed
        self.chunk_id_lock = asyncio.Lock()
        self.chunk_ids: list[Any] = []

    @classmethod
    async def create(cls, dimension: int, kvstore: KVStore | None = None, bank_id: str | None = None):
        instance = cls(dimension, kvstore, bank_id)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        if not self.kvstore:
            return

        index_key = f"{FAISS_INDEX_PREFIX}{self.bank_id}"
        stored_data = await self.kvstore.get(index_key)

        if stored_data:
            data = json.loads(stored_data)
            self.chunk_by_index = {int(k): Chunk.model_validate_json(v) for k, v in data["chunk_by_index"].items()}

            buffer = io.BytesIO(base64.b64decode(data["faiss_index"]))
            try:
                self.index = faiss.deserialize_index(np.load(buffer, allow_pickle=False))
                self.chunk_ids = [chunk.chunk_id for chunk in self.chunk_by_index.values()]
            except Exception as e:
                logger.debug(e, exc_info=True)
                raise ValueError(
                    "Error deserializing Faiss index from storage. If you recently upgraded your Llama Stack, Faiss, "
                    "or NumPy versions, you may need to delete the index and re-create it again or downgrade versions.\n"
                    f"The problematic index is stored in the key value store {self.kvstore} under the key '{index_key}'."
                ) from e

    async def _save_index(self):
        if not self.kvstore or not self.bank_id:
            return

        np_index = faiss.serialize_index(self.index)
        buffer = io.BytesIO()
        np.save(buffer, np_index, allow_pickle=False)
        data = {
            "chunk_by_index": {k: v.model_dump_json() for k, v in self.chunk_by_index.items()},
            "faiss_index": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        }

        index_key = f"{FAISS_INDEX_PREFIX}{self.bank_id}"
        await self.kvstore.set(key=index_key, value=json.dumps(data))

    async def delete(self):
        if not self.kvstore or not self.bank_id:
            return

        await self.kvstore.delete(f"{FAISS_INDEX_PREFIX}{self.bank_id}")

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        # Add dimension check
        embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        if embedding_dim != self.index.d:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.index.d}, got {embedding_dim}")

        indexlen = len(self.chunk_by_index)
        for i, chunk in enumerate(chunks):
            self.chunk_by_index[indexlen + i] = chunk

        async with self.chunk_id_lock:
            self.index.add(np.array(embeddings).astype(np.float32))
            self.chunk_ids.extend([chunk.chunk_id for chunk in chunks])

        # Save updated index
        await self._save_index()

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        if not set(chunk_ids).issubset(self.chunk_ids):
            return

        def remove_chunk(chunk_id: str):
            index = self.chunk_ids.index(chunk_id)
            self.index.remove_ids(np.array([index]))

            new_chunk_by_index = {}
            for idx, chunk in self.chunk_by_index.items():
                # Shift all chunks after the removed chunk to the left
                if idx > index:
                    new_chunk_by_index[idx - 1] = chunk
                else:
                    new_chunk_by_index[idx] = chunk
            self.chunk_by_index = new_chunk_by_index
            self.chunk_ids.pop(index)

        async with self.chunk_id_lock:
            for chunk_id in chunk_ids:
                remove_chunk(chunk_id)

        await self._save_index()

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        distances, indices = await asyncio.to_thread(self.index.search, embedding.reshape(1, -1).astype(np.float32), k)
        chunks = []
        scores = []
        for d, i in zip(distances[0], indices[0], strict=False):
            if i < 0:
                continue
            score = 1.0 / float(d) if d != 0 else float("inf")
            if score < score_threshold:
                continue
            chunks.append(self.chunk_by_index[int(i)])
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError(
            "Keyword search is not supported - underlying DB FAISS does not support this search mode"
        )

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError(
            "Hybrid search is not supported - underlying DB FAISS does not support this search mode"
        )


class FaissVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(self, config: FaissVectorIOConfig, inference_api: Inference, files_api: Files | None) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        self.config = config
        self.inference_api = inference_api
        self.cache: dict[str, VectorStoreWithIndex] = {}

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)
        # Load existing banks from kvstore
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store,
                await FaissIndex.create(vector_store.embedding_dimension, self.kvstore, vector_store.identifier),
                self.inference_api,
            )
            self.cache[vector_store.identifier] = index

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the inline faiss DB.
        This method is used by the Provider API to verify
        that the service is running correctly.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            vector_dimension = 128  # sample dimension
            faiss.IndexFlatL2(vector_dimension)
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Store in cache
        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store=vector_store,
            index=await FaissIndex.create(vector_store.embedding_dimension, self.kvstore, vector_store.identifier),
            inference_api=self.inference_api,
        )

    async def list_vector_stores(self) -> list[VectorStore]:
        return [i.vector_store for i in self.cache.values()]

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before unregistering vector stores.")

        if vector_store_id not in self.cache:
            return

        await self.cache[vector_store_id].index.delete()
        del self.cache[vector_store_id]
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=await FaissIndex.create(vector_store.embedding_dimension, self.kvstore, vector_store.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def insert_chunks(self, vector_store_id: str, chunks: list[Chunk], ttl_seconds: int | None = None) -> None:
        index = self.cache.get(vector_store_id)
        if index is None:
            raise ValueError(f"Vector DB {vector_store_id} not found. found: {self.cache.keys()}")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self, vector_store_id: str, query: InterleavedContent, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        index = self.cache.get(vector_store_id)
        if index is None:
            raise VectorStoreNotFoundError(vector_store_id)

        return await index.query_chunks(query, params)

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from a faiss index"""
        faiss_index = self.cache[store_id].index
        await faiss_index.delete_chunks(chunks_for_deletion)
