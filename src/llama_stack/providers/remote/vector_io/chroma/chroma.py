# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import json
from typing import Any
from urllib.parse import urlparse

import chromadb
from numpy.typing import NDArray

from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import VectorStoresProtocolPrivate
from llama_stack.providers.inline.vector_io.chroma import ChromaVectorIOConfig as InlineChromaVectorIOConfig
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex

from .config import ChromaVectorIOConfig as RemoteChromaVectorIOConfig

log = get_logger(name=__name__, category="vector_io::chroma")

ChromaClientType = chromadb.api.AsyncClientAPI | chromadb.api.ClientAPI

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:chroma:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:chroma:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:chroma:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:chroma:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:chroma:{VERSION}::"


# this is a helper to allow us to use async and non-async chroma clients interchangeably
async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class ChromaIndex(EmbeddingIndex):
    def __init__(self, client: ChromaClientType, collection, kvstore: KVStore | None = None):
        self.client = client
        self.collection = collection
        self.kvstore = kvstore

    async def initialize(self):
        pass

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        ids = [f"{c.metadata.get('document_id', '')}:{c.chunk_id}" for c in chunks]
        await maybe_await(
            self.collection.add(documents=[chunk.model_dump_json() for chunk in chunks], embeddings=embeddings, ids=ids)
        )

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        results = await maybe_await(
            self.collection.query(
                query_embeddings=[embedding.tolist()], n_results=k, include=["documents", "distances"]
            )
        )
        distances = results["distances"][0]
        documents = results["documents"][0]

        chunks = []
        scores = []
        for dist, doc in zip(distances, documents, strict=False):
            try:
                doc = json.loads(doc)
                chunk = Chunk(**doc)
            except Exception:
                log.exception(f"Failed to parse document: {doc}")
                continue

            score = 1.0 / float(dist) if dist != 0 else float("inf")
            if score < score_threshold:
                continue

            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        await maybe_await(self.client.delete_collection(self.collection.name))

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in Chroma")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete a single chunk from the Chroma collection by its ID."""
        ids = [f"{chunk.document_id}:{chunk.chunk_id}" for chunk in chunks_for_deletion]
        await maybe_await(self.collection.delete(ids=ids))

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in Chroma")


class ChromaVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: RemoteChromaVectorIOConfig | InlineChromaVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
    ) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        log.info(f"Initializing ChromaVectorIOAdapter with url: {config}")
        self.config = config
        self.inference_api = inference_api
        self.client = None
        self.cache = {}
        self.vector_store_table = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)

        if isinstance(self.config, RemoteChromaVectorIOConfig):
            log.info(f"Connecting to Chroma server at: {self.config.url}")
            url = self.config.url.rstrip("/")
            parsed = urlparse(url)

            if parsed.path and parsed.path != "/":
                raise ValueError("URL should not contain a path")

            self.client = await chromadb.AsyncHttpClient(host=parsed.hostname, port=parsed.port)
        else:
            log.info(f"Connecting to Chroma local db at: {self.config.db_path}")
            self.client = chromadb.PersistentClient(path=self.config.db_path)
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        collection = await maybe_await(
            self.client.get_or_create_collection(
                name=vector_store.identifier, metadata={"vector_store": vector_store.model_dump_json()}
            )
        )
        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store, ChromaIndex(self.client, collection), self.inference_api
        )

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id not in self.cache:
            log.warning(f"Vector DB {vector_store_id} not found")
            return

        await self.cache[vector_store_id].index.delete()
        del self.cache[vector_store_id]

    async def insert_chunks(self, vector_store_id: str, chunks: list[Chunk], ttl_seconds: int | None = None) -> None:
        index = await self._get_and_cache_vector_store_index(vector_store_id)
        if index is None:
            raise ValueError(f"Vector DB {vector_store_id} not found in Chroma")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self, vector_store_id: str, query: InterleavedContent, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(vector_store_id)

        if index is None:
            raise ValueError(f"Vector DB {vector_store_id} not found in Chroma")

        return await index.query_chunks(query, params)

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        # Try to load from kvstore
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise ValueError(f"Vector DB {vector_store_id} not found in Llama Stack")

        vector_store = VectorStore.model_validate_json(vector_store_data)
        collection = await maybe_await(self.client.get_collection(vector_store_id))
        if not collection:
            raise ValueError(f"Vector DB {vector_store_id} not found in Chroma")
        index = VectorStoreWithIndex(vector_store, ChromaIndex(self.client, collection), self.inference_api)
        self.cache[vector_store_id] = index
        return index

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from a Chroma vector store."""
        index = await self._get_and_cache_vector_store_index(store_id)
        if not index:
            raise ValueError(f"Vector DB {store_id} not found")

        await index.index.delete_chunks(chunks_for_deletion)
