# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
from typing import Any

from numpy.typing import NDArray
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker, WeightedRanker

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import VectorStoresProtocolPrivate
from llama_stack.providers.inline.vector_io.milvus import MilvusVectorIOConfig as InlineMilvusVectorIOConfig
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    RERANKER_TYPE_WEIGHTED,
    ChunkForDeletion,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from llama_stack.providers.utils.vector_io.vector_utils import sanitize_collection_name

from .config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig

logger = get_logger(name=__name__, category="vector_io::milvus")

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:milvus:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:milvus:{VERSION}::"


class MilvusIndex(EmbeddingIndex):
    def __init__(
        self, client: MilvusClient, collection_name: str, consistency_level="Strong", kvstore: KVStore | None = None
    ):
        self.client = client
        self.collection_name = sanitize_collection_name(collection_name)
        self.consistency_level = consistency_level
        self.kvstore = kvstore

    async def initialize(self):
        # MilvusIndex does not require explicit initialization
        # TODO: could move collection creation into initialization but it is not really necessary
        pass

    async def delete(self):
        if await asyncio.to_thread(self.client.has_collection, self.collection_name):
            await asyncio.to_thread(self.client.drop_collection, collection_name=self.collection_name)

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        if not await asyncio.to_thread(self.client.has_collection, self.collection_name):
            logger.info(f"Creating new collection {self.collection_name} with nullable sparse field")
            # Create schema for vector search
            schema = self.client.create_schema()
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,  # Enable text analysis for BM25
            )
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
            schema.add_field(field_name="chunk_content", datatype=DataType.JSON)
            # Add sparse vector field for BM25 (required by the function)
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

            # Create indexes
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
            # Add index for sparse field (required by BM25 function)
            index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

            # Add BM25 function for full-text search
            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["content"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

            await asyncio.to_thread(
                self.client.create_collection,
                self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self.consistency_level,
            )

        data = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "vector": embedding,
                    "chunk_content": chunk.model_dump(),
                    # sparse field will be handled by BM25 function automatically
                }
            )
        try:
            await asyncio.to_thread(self.client.insert, self.collection_name, data=data)
        except Exception as e:
            logger.error(f"Error inserting chunks into Milvus collection {self.collection_name}: {e}")
            raise e

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        search_res = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="vector",
            limit=k,
            output_fields=["*"],
            search_params={"params": {"radius": score_threshold}},
        )
        chunks = [Chunk(**res["entity"]["chunk_content"]) for res in search_res[0]]
        scores = [res["distance"] for res in search_res[0]]
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """
        Perform BM25-based keyword search using Milvus's built-in full-text search.
        """
        try:
            # Use Milvus's built-in BM25 search
            search_res = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                data=[query_string],  # Raw text query
                anns_field="sparse",  # Use sparse field for BM25
                output_fields=["chunk_content"],  # Output the chunk content
                limit=k,
                search_params={
                    "params": {
                        "drop_ratio_search": 0.2,  # Ignore low-importance terms
                    }
                },
            )

            chunks = []
            scores = []
            for res in search_res[0]:
                chunk = Chunk(**res["entity"]["chunk_content"])
                chunks.append(chunk)
                scores.append(res["distance"])  # BM25 score from Milvus

            # Filter by score threshold
            filtered_chunks = [chunk for chunk, score in zip(chunks, scores, strict=False) if score >= score_threshold]
            filtered_scores = [score for score in scores if score >= score_threshold]

            return QueryChunksResponse(chunks=filtered_chunks, scores=filtered_scores)

        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            # Fallback to simple text search
            return await self._fallback_keyword_search(query_string, k, score_threshold)

    async def _fallback_keyword_search(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """
        Fallback to simple text search when BM25 search is not available.
        """
        # Simple text search using content field
        search_res = await asyncio.to_thread(
            self.client.query,
            collection_name=self.collection_name,
            filter='content like "%{content}%"',
            filter_params={"content": query_string},
            output_fields=["*"],
            limit=k,
        )
        chunks = [Chunk(**res["chunk_content"]) for res in search_res]
        scores = [1.0] * len(chunks)  # Simple binary score for text search
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
        Hybrid search using Milvus's native hybrid search capabilities.

        This implementation uses Milvus's hybrid_search method which combines
        vector search and BM25 search with configurable reranking strategies.
        """
        search_requests = []

        # nprobe: Controls search accuracy vs performance trade-off
        # 10 balances these trade-offs for  RAG applications
        search_requests.append(
            AnnSearchRequest(data=[embedding.tolist()], anns_field="vector", param={"nprobe": 10}, limit=k)
        )

        # drop_ratio_search: Filters low-importance terms to improve search performance
        # 0.2 balances noise reduction with recall
        search_requests.append(
            AnnSearchRequest(data=[query_string], anns_field="sparse", param={"drop_ratio_search": 0.2}, limit=k)
        )

        if reranker_type == RERANKER_TYPE_WEIGHTED:
            alpha = (reranker_params or {}).get("alpha", 0.5)
            rerank = WeightedRanker(alpha, 1 - alpha)
        else:
            impact_factor = (reranker_params or {}).get("impact_factor", 60.0)
            rerank = RRFRanker(impact_factor)

        search_res = await asyncio.to_thread(
            self.client.hybrid_search,
            collection_name=self.collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=k,
            output_fields=["chunk_content"],
        )

        chunks = []
        scores = []
        for res in search_res[0]:
            chunk = Chunk(**res["entity"]["chunk_content"])
            chunks.append(chunk)
            scores.append(res["distance"])

        filtered_chunks = [chunk for chunk, score in zip(chunks, scores, strict=False) if score >= score_threshold]
        filtered_scores = [score for score in scores if score >= score_threshold]

        return QueryChunksResponse(chunks=filtered_chunks, scores=filtered_scores)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the Milvus collection."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        try:
            # Use IN clause with square brackets and single quotes for VARCHAR field
            chunk_ids_str = ", ".join(f"'{chunk_id}'" for chunk_id in chunk_ids)
            await asyncio.to_thread(
                self.client.delete, collection_name=self.collection_name, filter=f"chunk_id in [{chunk_ids_str}]"
            )
        except Exception as e:
            logger.error(f"Error deleting chunks from Milvus collection {self.collection_name}: {e}")
            raise


class MilvusVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: RemoteMilvusVectorIOConfig | InlineMilvusVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
    ) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        self.config = config
        self.cache = {}
        self.client = None
        self.inference_api = inference_api
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store,
                index=MilvusIndex(
                    client=self.client,
                    collection_name=vector_store.identifier,
                    consistency_level=self.config.consistency_level,
                    kvstore=self.kvstore,
                ),
                inference_api=self.inference_api,
            )
            self.cache[vector_store.identifier] = index
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            logger.info(f"Connecting to Milvus server at {self.config.uri}")
            self.client = MilvusClient(**self.config.model_dump(exclude_none=True))
        else:
            logger.info(f"Connecting to Milvus Lite at: {self.config.db_path}")
            uri = os.path.expanduser(self.config.db_path)
            self.client = MilvusClient(uri=uri)

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        self.client.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            consistency_level = "Strong"
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=MilvusIndex(self.client, vector_store.identifier, consistency_level=consistency_level),
            inference_api=self.inference_api,
        )

        self.cache[vector_store.identifier] = index

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
            index=MilvusIndex(client=self.client, collection_name=vector_store.identifier, kvstore=self.kvstore),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

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

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete a chunk from a milvus vector store."""
        index = await self._get_and_cache_vector_store_index(store_id)
        if not index:
            raise VectorStoreNotFoundError(store_id)

        await index.index.delete_chunks(chunks_for_deletion)
