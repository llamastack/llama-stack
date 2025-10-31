# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import heapq
from typing import Any

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import VectorStoresProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from llama_stack.providers.utils.vector_io.vector_utils import WeightedInMemoryAggregator, sanitize_collection_name

from .config import PGVectorVectorIOConfig

log = get_logger(name=__name__, category="vector_io::pgvector")

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:pgvector:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:pgvector:{VERSION}::"


def check_extension_version(cur):
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    return result[0] if result else None


def upsert_models(conn, keys_models: list[tuple[str, BaseModel]]):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = sql.SQL(
            """
            INSERT INTO metadata_store (key, data)
            VALUES %s
            ON CONFLICT (key) DO UPDATE
            SET data = EXCLUDED.data
        """
        )

        values = [(key, Json(model.model_dump())) for key, model in keys_models]
        execute_values(cur, query, values, template="(%s, %s)")


def load_models(cur, cls):
    cur.execute("SELECT key, data FROM metadata_store")
    rows = cur.fetchall()
    return [TypeAdapter(cls).validate_python(row["data"]) for row in rows]


class PGVectorIndex(EmbeddingIndex):
    # reference: https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
    PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION: dict[str, str] = {
        "L2": "<->",
        "L1": "<+>",
        "COSINE": "<=>",
        "INNER_PRODUCT": "<#>",
        "HAMMING": "<~>",
        "JACCARD": "<%>",
    }

    def __init__(
        self,
        vector_store: VectorStore,
        dimension: int,
        conn: psycopg2.extensions.connection,
        kvstore: KVStore | None = None,
        distance_metric: str = "COSINE",
    ):
        self.vector_store = vector_store
        self.dimension = dimension
        self.conn = conn
        self.kvstore = kvstore
        self.check_distance_metric_availability(distance_metric)
        self.distance_metric = distance_metric
        self.table_name = None

    async def initialize(self) -> None:
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Sanitize the table name by replacing hyphens with underscores
                # SQL doesn't allow hyphens in table names, and vector_store.identifier may contain hyphens
                # when created with patterns like "test-vector-db-{uuid4()}"
                sanitized_identifier = sanitize_collection_name(self.vector_store.identifier)
                self.table_name = f"vs_{sanitized_identifier}"

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        document JSONB,
                        embedding vector({self.dimension}),
                        content_text TEXT,
                        tokenized_content TSVECTOR
                    )
                """
                )

                # Create GIN index for full-text search performance
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_content_gin_idx
                    ON {self.table_name} USING GIN(tokenized_content)
                """
                )
        except Exception as e:
            log.exception(f"Error creating PGVectorIndex for vector_store: {self.vector_store.identifier}")
            raise RuntimeError(f"Error creating PGVectorIndex for vector_store: {self.vector_store.identifier}") from e

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        values = []
        for i, chunk in enumerate(chunks):
            content_text = interleaved_content_as_str(chunk.content)
            values.append(
                (
                    f"{chunk.chunk_id}",
                    Json(chunk.model_dump()),
                    embeddings[i].tolist(),
                    content_text,
                    content_text,  # Pass content_text twice - once for content_text column, once for to_tsvector function. Eg. to_tsvector(content_text) = tokenized_content
                )
            )

        query = sql.SQL(
            f"""
        INSERT INTO {self.table_name} (id, document, embedding, content_text, tokenized_content)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            document = EXCLUDED.document,
            content_text = EXCLUDED.content_text,
            tokenized_content = EXCLUDED.tokenized_content
    """
        )
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            execute_values(cur, query, values, template="(%s, %s, %s::vector, %s, to_tsvector('english', %s))")

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        """
        Performs vector similarity search using PostgreSQL's search function. Default distance metric is COSINE.

        Args:
            embedding: The query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            QueryChunksResponse with combined results
        """
        pgvector_search_function = self.get_pgvector_search_function()

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
            SELECT document, embedding {pgvector_search_function} %s::vector AS distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT %s
        """,
                (embedding.tolist(), k),
            )
            results = cur.fetchall()

            chunks = []
            scores = []
            for doc, dist in results:
                score = 1.0 / float(dist) if dist != 0 else float("inf")
                if score < score_threshold:
                    continue
                chunks.append(Chunk(**doc))
                scores.append(score)

            return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """
        Performs keyword-based search using PostgreSQL's full-text search with ts_rank scoring.

        Args:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            QueryChunksResponse with combined results
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Use plainto_tsquery to handle user input safely and ts_rank for relevance scoring
            cur.execute(
                f"""
            SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score
            FROM {self.table_name}
            WHERE tokenized_content @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """,
                (query_string, query_string, k),
            )
            results = cur.fetchall()

            chunks = []
            scores = []
            for doc, score in results:
                if score < score_threshold:
                    continue
                chunks.append(Chunk(**doc))
                scores.append(float(score))

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
        Hybrid search combining vector similarity and keyword search using configurable reranking.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker

        Returns:
            QueryChunksResponse with combined results
        """
        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods
        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

        # Convert responses to score dictionaries using chunk_id
        vector_scores = {
            chunk.chunk_id: score for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            chunk.chunk_id: score
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        # Combine scores using the reranking utility
        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type, reranker_params
        )

        # Efficient top-k selection because it only tracks the k best candidates it's seen so far
        top_k_items = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])

        # Filter by score threshold
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        # Create a map of chunk_id to chunk for both responses
        chunk_map = {c.chunk_id: c for c in vector_response.chunks + keyword_response.chunks}

        # Use the map to look up chunks by their IDs
        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the PostgreSQL table."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (chunk_ids))

    def get_pgvector_search_function(self) -> str:
        return self.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION[self.distance_metric]

    def check_distance_metric_availability(self, distance_metric: str) -> None:
        """Check if the distance metric is supported by PGVector.

        Args:
            distance_metric: The distance metric to check

        Raises:
            ValueError: If the distance metric is not supported
        """
        if distance_metric not in self.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION:
            supported_metrics = list(self.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION.keys())
            raise ValueError(
                f"Distance metric '{distance_metric}' is not supported by PGVector. "
                f"Supported metrics are: {', '.join(supported_metrics)}"
            )


class PGVectorVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self, config: PGVectorVectorIOConfig, inference_api: Inference, files_api: Files | None = None
    ) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        self.config = config
        self.inference_api = inference_api
        self.conn = None
        self.cache = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        log.info(f"Initializing PGVector memory adapter with config: {self.config}")
        self.kvstore = await kvstore_impl(self.config.persistence)
        await self.initialize_openai_vector_stores()

        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
            )
            self.conn.autocommit = True
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                version = check_extension_version(cur)
                if version:
                    log.info(f"Vector extension version: {version}")
                else:
                    raise RuntimeError("Vector extension is not installed.")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata_store (
                        key TEXT PRIMARY KEY,
                        data JSONB
                    )
                """
                )
        except Exception as e:
            log.exception("Could not connect to PGVector database server")
            raise RuntimeError("Could not connect to PGVector database server") from e

        # Load existing vector stores from KV store into cache
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            pgvector_index = PGVectorIndex(
                vector_store=vector_store,
                dimension=vector_store.embedding_dimension,
                conn=self.conn,
                kvstore=self.kvstore,
            )
            await pgvector_index.initialize()
            index = VectorStoreWithIndex(vector_store, index=pgvector_index, inference_api=self.inference_api)
            self.cache[vector_store.identifier] = index

    async def shutdown(self) -> None:
        if self.conn is not None:
            self.conn.close()
            log.info("Connection to PGVector database server closed")
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        # Persist vector DB metadata in the KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        # Save to kvstore for persistence
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Upsert model metadata in Postgres
        upsert_models(self.conn, [(vector_store.identifier, vector_store)])

        # Create and cache the PGVector index table for the vector DB
        pgvector_index = PGVectorIndex(
            vector_store=vector_store, dimension=vector_store.embedding_dimension, conn=self.conn, kvstore=self.kvstore
        )
        await pgvector_index.initialize()
        index = VectorStoreWithIndex(vector_store, index=pgvector_index, inference_api=self.inference_api)
        self.cache[vector_store.identifier] = index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        # Remove provider index and cache
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        # Delete vector DB metadata from KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before unregistering vector stores.")
        await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def insert_chunks(self, vector_store_id: str, chunks: list[Chunk], ttl_seconds: int | None = None) -> None:
        index = await self._get_and_cache_vector_store_index(vector_store_id)
        await index.insert_chunks(chunks)

    async def query_chunks(
        self, vector_store_id: str, query: InterleavedContent, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(vector_store_id)
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
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        index = PGVectorIndex(vector_store, vector_store.embedding_dimension, self.conn)
        await index.initialize()
        self.cache[vector_store_id] = VectorStoreWithIndex(vector_store, index, self.inference_api)
        return self.cache[vector_store_id]

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete a chunk from a PostgreSQL vector store."""
        index = await self._get_and_cache_vector_store_index(store_id)
        if not index:
            raise VectorStoreNotFoundError(store_id)

        await index.index.delete_chunks(chunks_for_deletion)
