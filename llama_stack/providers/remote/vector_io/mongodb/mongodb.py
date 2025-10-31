# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import heapq
import time
from typing import Any

from numpy.typing import NDArray
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
    VectorDBsProtocolPrivate,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import (
    OpenAIVectorStoreMixin,
)
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    EmbeddingIndex,
    VectorDBWithIndex,
)
from llama_stack.providers.utils.vector_io.vector_utils import (
    WeightedInMemoryAggregator,
    sanitize_collection_name,
)

from .config import MongoDBVectorIOConfig

logger = get_logger(name=__name__, category="vector_io::mongodb")

VERSION = "v1"
VECTOR_DBS_PREFIX = f"vector_dbs:mongodb:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:mongodb:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:mongodb:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:mongodb:{VERSION}::"


class MongoDBIndex(EmbeddingIndex):
    """MongoDB Atlas Vector Search index implementation optimized for RAG."""

    def __init__(
        self,
        vector_db: VectorDB,
        collection: Collection,
        config: MongoDBVectorIOConfig,
    ):
        self.vector_db = vector_db
        self.collection = collection
        self.config = config
        self.dimension = vector_db.embedding_dimension

    async def initialize(self) -> None:
        """Initialize the MongoDB collection and ensure vector search index exists."""
        try:
            # Create the collection if it doesn't exist
            collection_names = self.collection.database.list_collection_names()
            if self.collection.name not in collection_names:
                logger.info(f"Creating collection '{self.collection.name}'")
                # Create collection by inserting a dummy document
                dummy_doc = {"_id": "__dummy__", "dummy": True}
                self.collection.insert_one(dummy_doc)
                # Remove the dummy document
                self.collection.delete_one({"_id": "__dummy__"})
                logger.info(f"Collection '{self.collection.name}' created successfully")

            # Create optimized vector search index for RAG
            await self._create_vector_search_index()

            # Create text index for hybrid search
            await self._ensure_text_index()

        except Exception as e:
            logger.exception(
                f"Failed to initialize MongoDB index for vector_db: {self.vector_db.identifier}. "
                f"Collection name: {self.collection.name}. Error: {str(e)}"
            )
            # Don't fail completely - just log the error and continue
            logger.warning(
                "Continuing without complete index initialization. "
                "You may need to create indexes manually in MongoDB Atlas dashboard."
            )

    async def _create_vector_search_index(self) -> None:
        """Create optimized vector search index based on MongoDB RAG best practices."""
        try:
            # Check if vector search index exists
            indexes = list(self.collection.list_search_indexes())
            index_exists = any(idx.get("name") == self.config.index_name for idx in indexes)

            if not index_exists:
                # Create vector search index optimized for RAG
                # Based on MongoDB's RAG example using new vectorSearch format
                search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "numDimensions": self.dimension,
                                "path": self.config.path_field,
                                "similarity": self._convert_similarity_metric(self.config.similarity_metric),
                            }
                        ]
                    },
                    name=self.config.index_name,
                    type="vectorSearch",
                )

                logger.info(
                    f"Creating vector search index '{self.config.index_name}' for RAG on collection '{self.collection.name}'"
                )

                self.collection.create_search_index(model=search_index_model)

                # Wait for index to be ready (like in MongoDB RAG example)
                await self._wait_for_index_ready()

                logger.info("Vector search index created and ready for RAG queries")

        except Exception as e:
            logger.warning(f"Failed to create vector search index: {e}")

    def _convert_similarity_metric(self, metric: str) -> str:
        """Convert internal similarity metric to MongoDB Atlas format."""
        metric_map = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dotProduct": "dotProduct",
            "dot_product": "dotProduct",
        }
        return metric_map.get(metric, "cosine")

    async def _wait_for_index_ready(self) -> None:
        """Wait for the vector search index to be ready, based on MongoDB RAG example."""
        logger.info("Waiting for vector search index to be ready...")

        max_wait_time = 300  # 5 minutes max wait
        wait_interval = 5
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            try:
                indices = list(self.collection.list_search_indexes(self.config.index_name))
                if len(indices) and indices[0].get("queryable") is True:
                    logger.info(f"Vector search index '{self.config.index_name}' is ready for querying")
                    return
            except Exception:
                pass

            time.sleep(wait_interval)
            elapsed_time += wait_interval

        logger.warning(f"Vector search index may not be fully ready after {max_wait_time}s")

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray) -> None:
        """Add chunks with embeddings to MongoDB collection optimized for RAG."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}")

        documents = []
        for i, chunk in enumerate(chunks):
            # Structure document for optimal RAG retrieval
            doc = {
                "_id": chunk.chunk_id,
                "chunk_id": chunk.chunk_id,
                "text": interleaved_content_as_str(chunk.content),  # Key field for RAG context
                "content": interleaved_content_as_str(chunk.content),  # Backward compatibility
                "metadata": chunk.metadata or {},
                "chunk_metadata": (chunk.chunk_metadata.model_dump() if chunk.chunk_metadata else {}),
                self.config.path_field: embeddings[i].tolist(),  # Vector embeddings
                "document": chunk.model_dump(),  # Full chunk data
            }
            documents.append(doc)

        try:
            # Use upsert behavior for chunks
            for doc in documents:
                self.collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

            logger.debug(f"Successfully added {len(chunks)} chunks optimized for RAG to MongoDB collection")
        except Exception as e:
            logger.exception(f"Failed to add chunks to MongoDB collection: {e}")
            raise

    async def query_vector(
        self,
        embedding: NDArray,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """Perform vector similarity search optimized for RAG based on MongoDB example."""
        try:
            # Use MongoDB's vector search aggregation pipeline optimized for RAG
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.config.index_name,
                        "queryVector": embedding.tolist(),
                        "path": self.config.path_field,
                        "numCandidates": k * 10,  # Get more candidates for better results
                        "limit": k,
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "text": 1,  # Primary field for RAG context
                        "content": 1,  # Backward compatibility
                        "metadata": 1,
                        "chunk_metadata": 1,
                        "document": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
                {"$match": {"score": {"$gte": score_threshold}}},
            ]

            results = list(self.collection.aggregate(pipeline))

            chunks = []
            scores = []
            for result in results:
                score = result.get("score", 0.0)
                if score >= score_threshold:
                    chunk_data = result.get("document", {})
                    if chunk_data:
                        chunks.append(Chunk(**chunk_data))
                        scores.append(float(score))

            logger.debug(f"Vector search for RAG returned {len(chunks)} results")
            return QueryChunksResponse(chunks=chunks, scores=scores)

        except Exception as e:
            logger.exception(f"Vector search for RAG failed: {e}")
            raise RuntimeError(f"Vector search for RAG failed: {e}") from e

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """Perform text search using MongoDB's text search for RAG context retrieval."""
        try:
            # Ensure text index exists
            await self._ensure_text_index()

            pipeline = [
                {"$match": {"$text": {"$search": query_string}}},
                {
                    "$project": {
                        "_id": 0,
                        "text": 1,  # Primary field for RAG context
                        "content": 1,  # Backward compatibility
                        "metadata": 1,
                        "chunk_metadata": 1,
                        "document": 1,
                        "score": {"$meta": "textScore"},
                    }
                },
                {"$match": {"score": {"$gte": score_threshold}}},
                {"$sort": {"score": {"$meta": "textScore"}}},
                {"$limit": k},
            ]

            results = list(self.collection.aggregate(pipeline))

            chunks = []
            scores = []
            for result in results:
                score = result.get("score", 0.0)
                if score >= score_threshold:
                    chunk_data = result.get("document", {})
                    if chunk_data:
                        chunks.append(Chunk(**chunk_data))
                        scores.append(float(score))

            logger.debug(f"Keyword search for RAG returned {len(chunks)} results")
            return QueryChunksResponse(chunks=chunks, scores=scores)

        except Exception as e:
            logger.exception(f"Keyword search for RAG failed: {e}")
            raise RuntimeError(f"Keyword search for RAG failed: {e}") from e

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Perform hybrid search for enhanced RAG context retrieval."""
        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods
        vector_response = await self.query_vector(embedding, k, 0.0)
        keyword_response = await self.query_keyword(query_string, k, 0.0)

        # Convert responses to score dictionaries
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

        # Get top-k results
        top_k_items = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])

        # Filter by score threshold
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        # Create chunk map
        chunk_map = {c.chunk_id: c for c in vector_response.chunks + keyword_response.chunks}

        # Build final results
        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        logger.debug(f"Hybrid search for RAG returned {len(chunks)} results")
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from MongoDB collection."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        try:
            result = self.collection.delete_many({"_id": {"$in": chunk_ids}})
            logger.debug(f"Deleted {result.deleted_count} chunks from MongoDB collection")
        except Exception as e:
            logger.exception(f"Failed to delete chunks: {e}")
            raise

    async def delete(self) -> None:
        """Delete the entire collection."""
        try:
            self.collection.drop()
            logger.debug(f"Dropped MongoDB collection: {self.collection.name}")
        except Exception as e:
            logger.exception(f"Failed to drop collection: {e}")
            raise

    async def _ensure_text_index(self) -> None:
        """Ensure text search index exists on content fields for RAG."""
        try:
            indexes = list(self.collection.list_indexes())
            text_index_exists = any(
                any(key.startswith(("content", "text")) for key in idx.get("key", {}).keys())
                and idx.get("textIndexVersion") is not None
                for idx in indexes
            )

            if not text_index_exists:
                logger.info("Creating text search index on content fields for RAG")
                # Index both 'text' and 'content' fields for comprehensive text search
                self.collection.create_index([("text", "text"), ("content", "text")])
                logger.info("Text search index created successfully for RAG")

        except Exception as e:
            logger.warning(f"Failed to create text index for RAG: {e}")


class MongoDBVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    """MongoDB Atlas Vector Search adapter for Llama Stack optimized for RAG workflows."""

    def __init__(
        self,
        config: MongoDBVectorIOConfig,
        inference_api,
        files_api=None,
        models_api=None,
    ) -> None:
        # Handle the case where files_api might be a ProviderSpec that needs resolution
        resolved_files_api = files_api
        super().__init__(files_api=resolved_files_api, kvstore=None)
        self.config = config
        self.inference_api = inference_api
        self.models_api = models_api
        self.client: MongoClient | None = None
        self.database: Database | None = None
        self.cache: dict[str, VectorDBWithIndex] = {}
        self.kvstore: KVStore | None = None

    async def initialize(self) -> None:
        """Initialize MongoDB connection optimized for RAG workflows."""
        logger.info("Initializing MongoDB Atlas Vector IO adapter for RAG")

        try:
            # Initialize KV store for metadata
            self.kvstore = await kvstore_impl(self.config.kvstore)

            # Connect to MongoDB with optimized settings for RAG
            self.client = MongoClient(
                self.config.connection_string,
                server_api=ServerApi("1"),
                maxPoolSize=self.config.max_pool_size,
                serverSelectionTimeoutMS=self.config.timeout_ms,
                # Additional settings for RAG performance
                retryWrites=True,
                readPreference="primaryPreferred",
            )

            # Test connection
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB Atlas for RAG")

            # Get database
            self.database = self.client[self.config.database_name]

            # Initialize OpenAI vector stores
            await self.initialize_openai_vector_stores()

            # Load existing vector databases
            await self._load_existing_vector_dbs()

            logger.info("MongoDB Atlas Vector IO adapter for RAG initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize MongoDB Atlas Vector IO adapter for RAG")
            raise RuntimeError("Failed to initialize MongoDB Atlas Vector IO adapter for RAG") from e

    async def shutdown(self) -> None:
        """Shutdown MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB Atlas RAG connection closed")

    async def health(self) -> HealthResponse:
        """Perform health check on MongoDB connection."""
        try:
            if self.client:
                self.client.admin.command("ping")
                return HealthResponse(status=HealthStatus.OK)
            else:
                return HealthResponse(status=HealthStatus.ERROR, message="MongoDB client not initialized")
        except Exception as e:
            return HealthResponse(
                status=HealthStatus.ERROR,
                message=f"MongoDB RAG health check failed: {str(e)}",
            )

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        """Register a new vector database optimized for RAG."""
        if self.database is None:
            raise RuntimeError("MongoDB database not initialized")

        # Create collection name from vector DB identifier
        collection_name = sanitize_collection_name(vector_db.identifier)
        collection = self.database[collection_name]

        # Create and initialize MongoDB index optimized for RAG
        mongodb_index = MongoDBIndex(vector_db, collection, self.config)
        await mongodb_index.initialize()

        # Create vector DB with index wrapper
        vector_db_with_index = VectorDBWithIndex(
            vector_db=vector_db,
            index=mongodb_index,
            inference_api=self.inference_api,
        )

        # Cache the vector DB
        self.cache[vector_db.identifier] = vector_db_with_index

        # Save vector database info to KVStore for persistence
        if self.kvstore:
            await self.kvstore.set(
                f"{VECTOR_DBS_PREFIX}{vector_db.identifier}",
                vector_db.model_dump_json(),
            )

        logger.info(f"Registered vector database for RAG: {vector_db.identifier}")

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        """Unregister a vector database."""
        if vector_db_id in self.cache:
            await self.cache[vector_db_id].index.delete()
            del self.cache[vector_db_id]

        # Clean up from KV store
        if self.kvstore:
            await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_db_id}")

        logger.info(f"Unregistered vector database: {vector_db_id}")

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert chunks into the vector database optimized for RAG."""
        vector_db_with_index = await self._get_vector_db_index(vector_db_id)
        await vector_db_with_index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Query chunks from the vector database optimized for RAG context retrieval."""
        vector_db_with_index = await self._get_vector_db_index(vector_db_id)
        return await vector_db_with_index.query_chunks(query, params)

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from the vector database."""
        vector_db_with_index = await self._get_vector_db_index(store_id)
        await vector_db_with_index.index.delete_chunks(chunks_for_deletion)

    async def _get_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex:
        """Get vector database index from cache."""
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        raise VectorStoreNotFoundError(vector_db_id)

    async def _load_existing_vector_dbs(self) -> None:
        """Load existing vector databases from KVStore."""
        if not self.kvstore:
            return

        try:
            # Use keys_in_range to get all vector database keys from KVStore
            # This searches for keys with the prefix by using range scan
            start_key = VECTOR_DBS_PREFIX
            # Create an end key by incrementing the last character
            end_key = VECTOR_DBS_PREFIX[:-1] + chr(ord(VECTOR_DBS_PREFIX[-1]) + 1)

            vector_db_keys = await self.kvstore.keys_in_range(start_key, end_key)

            for key in vector_db_keys:
                try:
                    vector_db_data = await self.kvstore.get(key)
                    if vector_db_data:
                        import json

                        vector_db = VectorDB(**json.loads(vector_db_data))
                        # Register the vector database without re-initializing
                        await self._register_existing_vector_db(vector_db)
                        logger.info(f"Loaded existing RAG-optimized vector database: {vector_db.identifier}")
                except Exception as e:
                    logger.warning(f"Failed to load vector database from key {key}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to load existing vector databases: {e}")

    async def _register_existing_vector_db(self, vector_db: VectorDB) -> None:
        """Register an existing vector database without re-initialization."""
        if self.database is None:
            raise RuntimeError("MongoDB database not initialized")

        # Create collection name from vector DB identifier
        collection_name = sanitize_collection_name(vector_db.identifier)
        collection = self.database[collection_name]

        # Create MongoDB index without initialization (collection already exists)
        mongodb_index = MongoDBIndex(vector_db, collection, self.config)

        # Create vector DB with index wrapper
        vector_db_with_index = VectorDBWithIndex(
            vector_db=vector_db,
            index=mongodb_index,
            inference_api=self.inference_api,
        )

        # Cache the vector DB
        self.cache[vector_db.identifier] = vector_db_with_index
