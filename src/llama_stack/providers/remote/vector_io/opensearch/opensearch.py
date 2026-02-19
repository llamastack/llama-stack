# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from typing import Any, List, Optional

from llama_stack_api import (
    ChunkForDeletion,
    DeleteChunksRequest,
    EmbeddedChunk,
    Files,
    Inference,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorIO,
    VectorStore,
    VectorStoreNotFoundError,
)
from llama_stack_api.shared.schemas import HealthResponse, HealthStatus, StackComponentConfig
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.openai_vector_store_mixin import (
    OpenAIVectorStoreMixin,
)
from llama_stack.providers.utils.vector_io.vector_utils import (
    load_embedded_chunk_with_backward_compat,
)
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorStoreWithIndex,
)

try:
    from opensearchpy import OpenSearch, helpers
except ImportError:
    OpenSearch = None
    helpers = None

from .config import OpenSearchVectorIOConfig

logger = get_logger(name=__name__, category="vector_io::opensearch")


class OpenSearchIndex(EmbeddingIndex):
    def __init__(self, client: Any, index_name: str, dimension: int):
        self.client = client
        self.index_name = index_name.lower()  # OpenSearch indices must be lowercase
        self.dimension = dimension

    async def initialize(self):
        # Check if index exists
        exists = await asyncio.to_thread(self.client.indices.exists, index=self.index_name)
        if not exists:
            # Create index with k-NN mapping
            mapping = {
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",  # or "faiss" or "nmslib", lucene is standard
                                "space_type": "l2",
                            },
                        },
                        "chunk_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        # We store full JSON in a keyword or text field to retrieve it back
                        "chunk_content": {"type": "keyword", "index": False},
                        "metadata": {"type": "object"},
                    }
                },
            }
            try:
                await asyncio.to_thread(
                    self.client.indices.create, index=self.index_name, body=mapping
                )
            except Exception as e:
                # Handle race condition where index might be created concurrently
                logger.warning(f"Index creation failed (might already exist): {e}")

    async def delete(self):
        try:
            await asyncio.to_thread(self.client.indices.delete, index=self.index_name, ignore=[404])
        except Exception as e:
            logger.error(f"Failed to delete index {self.index_name}: {e}")

    async def add_chunks(self, chunks: List[EmbeddedChunk]):
        actions = []
        for chunk in chunks:
            doc = {
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "embedding": chunk.embedding,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_content": chunk.model_dump_json(),
            }
            actions.append(doc)

        # helpers.bulk is synchronous, run in thread
        success, failed = await asyncio.to_thread(
            helpers.bulk, self.client, actions, refresh=True
        )
        if failed:
            logger.error(f"Failed to index {len(failed)} documents in {self.index_name}")

    async def query_vector(
        self,
        embedding: List[float],
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": k,
                    }
                }
            },
        }

        response = await asyncio.to_thread(
            self.client.search, index=self.index_name, body=query
        )

        chunks = []
        scores = []
        for hit in response["hits"]["hits"]:
            # OpenSearch L2 score is 1 / (1 + l2_squared)
            score = hit["_score"]
            if score < score_threshold:
                continue
            
            source = hit["_source"]
            chunk = self._reconstruct_chunk(source)
            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        query = {
            "size": k,
            "query": {
                "match": {
                    "content": query_string
                }
            },
        }
        
        response = await asyncio.to_thread(
            self.client.search, index=self.index_name, body=query
        )
        
        chunks = []
        scores = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            if score < score_threshold:
                continue
            
            source = hit["_source"]
            chunk = self._reconstruct_chunk(source)
            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: List[float],
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        # Simple hybrid using compound bool query
        query = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": k,
                                    "boost": 0.5 
                                }
                            }
                        },
                        {
                            "match": {
                                "content": {
                                    "query": query_string,
                                    "boost": 0.5
                                }
                            }
                        }
                    ]
                }
            }
        }

        response = await asyncio.to_thread(
            self.client.search, index=self.index_name, body=query
        )
        
        chunks = []
        scores = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            if score < score_threshold:
                continue
            
            source = hit["_source"]
            chunk = self._reconstruct_chunk(source)
            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: List[ChunkForDeletion]):
        actions = []
        for chunk in chunks_for_deletion:
             actions.append({
                "_op_type": "delete",
                "_index": self.index_name,
                "_id": chunk.chunk_id,
            })
        
        # We ignore 404s (not found) during bulk delete to be safe
        await asyncio.to_thread(
             helpers.bulk, self.client, actions, raise_on_error=False, refresh=True
        )

    def _reconstruct_chunk(self, source: dict) -> EmbeddedChunk:
        chunk_content_json = source.get("chunk_content")
        if chunk_content_json:
             # It was stored as JSON string
            import json
            if isinstance(chunk_content_json, str):
                data = json.loads(chunk_content_json)
                return load_embedded_chunk_with_backward_compat(data)
            else:
                # Provide fallback if it was somehow stored as object
                return ChunkForDeletion(
                    chunk_id=source["chunk_id"]
                ) # This shouldn't happen with our add_chunks logic

        # Fallback reconstruction
        return EmbeddedChunk(
            chunk_id=source["chunk_id"],
            content=source["content"],
            embedding=source["embedding"],
            metadata=source.get("metadata", {}),
        )


class OpenSearchVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO):
    def __init__(
        self,
        config: OpenSearchVectorIOConfig,
        inference_api: Inference,
        files_api: Optional[Files] = None,
    ) -> None:
        super().__init__(inference_api=inference_api, files_api=files_api, kvstore=None)
        self.config = config
        self.client = None
        self.cache = {}

    async def initialize(self) -> None:
        if OpenSearch is None:
            raise ImportError(
                "opensearch-py is not installed. Please install it with `pip install opensearch-py`."
            )
        
        auth = None
        if self.config.username and self.config.password:
             auth = (self.config.username, self.config.password.get_secret_value())

        self.client = OpenSearch(
            hosts=[{"host": self.config.host, "port": self.config.port}],
            http_compress=True,
            http_auth=auth,
            use_ssl=self.config.use_ssl,
            verify_certs=self.config.verify_certs,
        )
        
        # Try to ping or get info to verify connection
        try:
            await asyncio.to_thread(self.client.info)
        except Exception as e:
            logger.warning(f"Could not connect to OpenSearch at startup: {e}")
        
        await self.initialize_openai_vector_stores()

    async def health(self) -> HealthResponse:
        try:
             await asyncio.to_thread(self.client.ping)
             return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
             return HealthResponse(
                status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"
            )

    async def shutdown(self) -> None:
        if self.client:
            self.client.close()
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        index = OpenSearchIndex(self.client, vector_store.identifier, vector_store.embedding_dimension)
        # Ensure index exists
        await index.initialize()
        
        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store=vector_store,
            index=index,
            inference_api=self.inference_api
        )

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

    async def list_vector_stores(self) -> List[VectorStore]:
        return [item.vector_store for item in self.cache.values()]

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        store = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not store:
            raise VectorStoreNotFoundError(request.vector_store_id)
        
        await store.index.add_chunks(request.chunks)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        store = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not store:
            raise VectorStoreNotFoundError(request.vector_store_id)
        
        return await store.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        store = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not store:
            raise VectorStoreNotFoundError(request.vector_store_id)
             
        await store.index.delete_chunks(request.chunks)

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> Optional[VectorStoreWithIndex]:
        # For now, only return from cache.
        # Capability to "load" from existing OpenSearch index not explicitly in cache is a TODO
        # but requires knowing dimension and metadata which we don't persist separately yet.
        return self.cache.get(vector_store_id)
