# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from elasticsearch import ApiError, AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from numpy.typing import NDArray

from llama_stack.core.storage.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from llama_stack_api import (
    EmbeddedChunk,
    Files,
    Inference,
    InterleavedContent,
    QueryChunksResponse,
    VectorIO,
    VectorStore,
    VectorStoreNotFoundError,
    VectorStoresProtocolPrivate,
)

from .config import ElasticsearchVectorIOConfig

log = get_logger(name=__name__, category="vector_io::elasticsearch")

# KV store prefixes for vector databases
VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:elasticsearch:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:elasticsearch:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:elasticsearch:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:elasticsearch:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:elasticsearch:{VERSION}::"


class ElasticsearchIndex(EmbeddingIndex):
    def __init__(self, client: AsyncElasticsearch, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    # Check if the rerank_params contains the following structure:
    # {
    #   "retrievers": {
    #       "standard": {"weight": 0.7},
    #       "knn": {"weight": 0.3}
    #   }
    # }
    async def _is_rerank_linear_param_valid(self, value: dict) -> bool:
        """Validate linear reranker parameters structure."""
        try:
            retrievers = value.get("retrievers", {})
            return (
                isinstance(retrievers.get("standard"), dict)
                and isinstance(retrievers.get("knn"), dict)
                and "weight" in retrievers["standard"]
                and "weight" in retrievers["knn"]
            )
        except (AttributeError, TypeError):
            return False

    async def initialize(self) -> None:
        # Elasticsearch collections (indexes) are created on-demand in add_chunks
        # If the index does not exist, it will be created in add_chunks.
        pass

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        """Adds chunks to the Elasticsearch index."""
        if not chunks:
            return

        try:
            await self.client.indices.create(
                index=self.collection_name,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "chunk_id": {"type": "keyword"},
                            "metadata": {"type": "object"},
                            "chunk_metadata": {"type": "object"},
                            "embedding": {"type": "dense_vector", "dims": len(chunks[0].embedding)},
                            "embedding_dimension": {"type": "integer"},
                            "embedding_model": {"type": "keyword"},
                        }
                    }
                },
            )
        except ApiError as e:
            if e.status_code != 400 or "resource_already_exists_exception" not in e.message:
                log.error(f"Error creating Elasticsearch index {self.collection_name}: {e}")
                raise

        actions = []
        for chunk in chunks:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.collection_name,
                    "_id": chunk.chunk_id,
                    "_source": chunk.model_dump(
                        exclude_none=True,
                        include={
                            "content",
                            "chunk_id",
                            "metadata",
                            "chunk_metadata",
                            "embedding",
                            "embedding_dimension",
                            "embedding_model",
                        },
                    ),
                }
            )

        try:
            successful_count, error_count = await async_bulk(
                client=self.client, actions=actions, timeout="300s", refresh=True, raise_on_error=False, stats_only=True
            )
            if error_count > 0:
                log.warning(
                    f"{error_count} out of {len(chunks)} documents failed to upload in Elasticsearch index {self.collection_name}"
                )

            log.info(f"Successfully added {successful_count} chunks to Elasticsearch index {self.collection_name}")
        except Exception as e:
            log.error(f"Error adding chunks to Elasticsearch index {self.collection_name}: {e}")
            raise

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the Elasticsearch index."""

        actions = []
        for chunk in chunks_for_deletion:
            actions.append({"_op_type": "delete", "_index": self.collection_name, "_id": chunk.chunk_id})

        try:
            successful_count, error_count = await async_bulk(
                client=self.client, actions=actions, timeout="300s", refresh=True, raise_on_error=True, stats_only=True
            )
            if error_count > 0:
                log.warning(
                    f"{error_count} out of {len(chunks_for_deletion)} documents failed to be deleted in Elasticsearch index {self.collection_name}"
                )

            log.info(f"Successfully deleted {successful_count} chunks from Elasticsearch index {self.collection_name}")
        except Exception as e:
            log.error(f"Error deleting chunks from Elasticsearch index {self.collection_name}: {e}")
            raise

    async def _results_to_chunks(self, results: dict) -> QueryChunksResponse:
        """Convert search results to QueryChunksResponse."""

        chunks, scores = [], []
        for result in results.get("hits", {}).get("hits", []):
            try:
                source = result.get("_source", {})
                chunk = EmbeddedChunk(
                    content=source.get("content"),
                    chunk_id=result.get("_id"),
                    embedding=source.get("embedding", []),
                    embedding_dimension=source.get("embedding_dimension", len(source.get("embedding", []))),
                    embedding_model=source.get("embedding_model", "unknown"),
                    chunk_metadata=source.get("chunk_metadata", {}),
                    metadata=source.get("metadata", {}),
                )
            except Exception:
                log.exception("Failed to parse chunk")
                continue

            chunks.append(chunk)
            scores.append(result.get("_score"))

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        """Vector search using kNN."""

        try:
            results = await self.client.search(
                index=self.collection_name,
                query={"knn": {"field": "embedding", "query_vector": embedding.tolist(), "k": k}},
                min_score=score_threshold,
                size=k,
                source={"exclude_vectors": False},  # Retrieve the embedding
                ignore_unavailable=True,  # In case the index does not exist
            )
        except Exception as e:
            log.error(f"Error performing vector query on Elasticsearch index {self.collection_name}: {e}")
            raise

        return await self._results_to_chunks(results)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """Keyword search using match query."""

        try:
            results = await self.client.search(
                index=self.collection_name,
                query={"match": {"content": {"query": query_string}}},
                min_score=score_threshold,
                size=k,
                source={"exclude_vectors": False},  # Retrieve the embedding
                ignore_unavailable=True,  # In case the index does not exist
            )
        except Exception as e:
            log.error(f"Error performing keyword query on Elasticsearch index {self.collection_name}: {e}")
            raise

        return await self._results_to_chunks(results)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        supported_retrievers = ["rrf", "linear"]
        if reranker_type not in supported_retrievers:
            raise ValueError(f"Unsupported reranker type: {reranker_type}. Supported types are: {supported_retrievers}")

        retriever = {
            reranker_type: {
                "retrievers": [
                    {"retriever": {"standard": {"query": {"match": {"content": query_string}}}}},
                    {"retriever": {"knn": {"field": "embedding", "query_vector": embedding.tolist(), "k": k}}},
                ]
            }
        }

        # Add reranker parameters if provided for RRF (e.g. rank_constant, rank_window_size, filter)
        # see https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever
        if reranker_type == "rrf" and reranker_params is not None:
            allowed_rrf_params = {"rank_constant", "rank_windows_size", "filter"}
            extra_keys = set(reranker_params.keys()) - allowed_rrf_params
            if extra_keys:
                raise ValueError(
                    f"Unsupported RRF reranker parameters: {extra_keys}. Allowed parameters are: {allowed_rrf_params}"
                )
            retriever["rrf"].update(reranker_params)
        elif reranker_type == "linear" and reranker_params is not None:
            # Add reranker parameters (i.e. weights) for linear
            # see https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever
            if await self._is_rerank_linear_param_valid(reranker_params) is False:
                raise ValueError(
                    "Invalid linear reranker parameters. Expected structure: "
                    '{ "retrievers": { "standard": {"weight": float}, "knn": {"weight": float} } }'
                )
            try:
                # The first retriever is standard (see retriever variable above)
                retriever["linear"]["retrievers"][0].update(reranker_params["retrievers"]["standard"])
                # The second retriever is knn (see retriever variable above)
                retriever["linear"]["retrievers"][1].update(reranker_params["retrievers"]["knn"])
            except Exception as e:
                log.error(f"Error updating linear retrievers parameters: {e}")
                raise
        try:
            results = await self.client.search(
                index=self.collection_name,
                size=k,
                retriever=retriever,
                min_score=score_threshold,
                source={"exclude_vectors": False},  # Retrieve the embedding
                ignore_unavailable=True,  # In case the index does not exist
            )
        except Exception as e:
            log.error(f"Error performing hybrid query on Elasticsearch index {self.collection_name}: {e}")
            raise

        return await self._results_to_chunks(results)

    async def delete(self):
        """Delete the entire Elasticsearch index with collection_name."""

        try:
            await self.client.indices.delete(index=self.collection_name, ignore_unavailable=True)
        except Exception as e:
            log.error(f"Error deleting Elasticsearch index {self.collection_name}: {e}")
            raise


class ElasticsearchVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: ElasticsearchVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
    ) -> None:
        super().__init__(inference_api=inference_api, files_api=files_api, kvstore=None)
        self.config = config
        self.client: AsyncElasticsearch = None
        self.cache = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        self.client = AsyncElasticsearch(hosts=self.config.elasticsearch_url, api_key=self.config.elasticsearch_api_key)
        self.kvstore = await kvstore_impl(self.config.persistence)

        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store, ElasticsearchIndex(self.client, vector_store.identifier), self.inference_api
            )
            self.cache[vector_store.identifier] = index
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        await self.client.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        assert self.kvstore is not None
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=ElasticsearchIndex(self.client, vector_store.identifier),
            inference_api=self.inference_api,
        )

        self.cache[vector_store.identifier] = index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        assert self.kvstore is not None
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        if self.vector_store_table is None:
            raise ValueError(f"Vector DB not found {vector_store_id}")

        vector_store = await self.vector_store_table.get_vector_store(vector_store_id)
        if not vector_store:
            raise VectorStoreNotFoundError(vector_store_id)

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=ElasticsearchIndex(client=self.client, collection_name=vector_store.identifier),
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

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from an Elasticsearch vector store."""
        index = await self._get_and_cache_vector_store_index(store_id)
        if not index:
            raise ValueError(f"Vector DB {store_id} not found")

        await index.index.delete_chunks(chunks_for_deletion)
