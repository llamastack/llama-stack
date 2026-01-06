# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from array import array
from typing import Any

from numpy.typing import NDArray
import oracledb

from llama_stack.core.storage.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin, VERSION as OpenAIMixinVersion
from llama_stack.providers.utils.memory.vector_store import (
    RERANKER_TYPE_WEIGHTED,
    ChunkForDeletion,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from llama_stack.providers.utils.vector_io.vector_utils import sanitize_collection_name
from llama_stack_api import (
    Chunk,
    Files,
    Inference,
    InterleavedContent,
    QueryChunksResponse,
    VectorIO,
    VectorStore,
    VectorStoreNotFoundError,
    VectorStoresProtocolPrivate,
)
from llama_stack_api.internal.kvstore import KVStore

from llama_stack.providers.remote.vector_io.oci.config import OCI26aiVectorIOConfig

logger = get_logger(name=__name__, category="vector_io::oci26ai")

VERSION = "v1"
VECTOR_DBS_PREFIX = f"vector_stores:oci26ai:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:oci26ai:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:oci26ai:{OpenAIMixinVersion}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:oci26ai:{OpenAIMixinVersion}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:oci26ai:{VERSION}::"


class OCI26aiIndex(EmbeddingIndex):
    def __init__(
        self,
        connection,
        vector_store: VectorStore, 
        consistency_level="Strong",
        kvstore: KVStore | None = None,
    ):
        self.connection = connection
        self.vector_store = vector_store
        self.table_name = sanitize_collection_name(vector_store.vector_store_name)
        self.identifier = vector_store.vector_store_id
        self.dimensions = vector_store.embedding_dimension
        self.consistency_level = consistency_level
        self.kvstore = kvstore
    
    async def initialize(self) -> None:
        logger.info(f"Attempting to create table: {self.table_name}")
        cursor = self.connection.cursor()
        try:
            #  Create table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    chunk_id VARCHAR2(100) PRIMARY KEY,
                    content CLOB,
                    vector VECTOR({self.dimensions}, FLOAT32),
                    metadata JSON,
                    chunk_metadata JSON
                );
            """
            logger.debug(f"Executing SQL: {create_table_sql}")
            cursor.execute(create_table_sql)
            logger.info(f"Table {self.table_name} created successfully")

            await self.create_indexes()
        finally:
            cursor.close()

    async def index_exists(self, index_name: str) -> bool:
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM USER_INDEXES 
                WHERE INDEX_NAME = :index_name
            """, index_name=index_name.upper())
            count, = cursor.fetchone()
            return count > 0
        finally:
            cursor.close()


    async def create_indexes(self):
        indexes = [
            {
                "name": f"{self.table_name}_content_idx",
                "sql": f"""
                    CREATE INDEX RRILEY_TEST_CONTENT_IDX
                    ON rriley_test(content)
                    INDEXTYPE IS CTXSYS.CONTEXT 
                    PARAMETERS ('SYNC (EVERY "FREQ=SECONDLY;INTERVAL=5")');
                """
            },
            {
                "name": f"{self.table_name}_vector_ivf_idx",
                "sql": f"""
                    CREATE VECTOR INDEX {self.table_name}_vector_ivf_idx
                    ON {self.table_name}(vector)
                    ORGANIZATION NEIGHBOR PARTITIONS
                    DISTANCE COSINE
                    WITH TARGET ACCURACY 95
                """
            }
        ]

        for idx in indexes:
            if not await self.index_exists(idx["name"]):
                logger.info(f"Creating index: {idx['name']}")
                cursor = self.connection.cursor()
                try:
                    cursor.execute(idx['sql'])
                    logger.info(f"Index {idx['name']} created successfully")
                finally:
                    cursor.close()
            else:
                logger.info(f"Index {idx['name']} already exists, skipping")

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        data = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk_step = chunk.model_dump()
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "vector": array("f", embedding),
                    "metadata": json.dumps(chunk_step.get('metadata')),
                    "chunk_metadata": json.dumps(chunk_step.get('chunk_metadata'))
                }
            )
        cursor = self.connection.cursor()
        try:
            query = f"""
                MERGE INTO {self.table_name} t
                USING (
                    SELECT
                        :chunk_id       AS chunk_id,
                        :content        AS content,
                        :vector         AS vector,
                        :metadata       AS metadata,
                        :chunk_metadata AS chunk_metadata
                    FROM dual
                ) s
                ON (t.chunk_id = s.chunk_id)

                WHEN MATCHED THEN
                UPDATE SET
                    t.content           = s.content,
                    t.vector            = TO_VECTOR(s.vector),
                    t.metadata          = s.metadata,
                    t.chunk_metadata    = s.chunk_metadata

                WHEN NOT MATCHED THEN
                INSERT (chunk_id, content, vector, metadata, chunk_metadata)
                VALUES (s.chunk_id, s.content, TO_VECTOR(s.vector), s.metadata, s.chunk_metadata)
                """
            logger.debug(f"query: {query}")
            cursor.executemany(
                query,
                data,
            )
            logger.info('Merge completed successfully')
        except Exception as e:
            logger.error(f"Error inserting chunks into Oracle 26AI table {self.table_name}: {e}")
            raise
        finally: 
            cursor.close()

    async def query_vector(
        self,
        embedding: NDArray,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """
        Oracle vector search using literal vector string and COSINE similarity.
        Returns top-k chunks and their similarity scores.
        """
        cursor = self.connection.cursor()
        query_vector = array("f", embedding.astype("float32"))
        query = f"""
            SELECT
                content, chunk_id, metadata, chunk_metadata, vector,
                1 - VECTOR_DISTANCE(:query_vector, vector, COSINE) AS similarity
            FROM {self.table_name}
            WHERE 1 - VECTOR_DISTANCE(:query_vector, vector, COSINE) >= :score_threshold
            ORDER BY similarity DESC FETCH FIRST :k ROWS ONLY
        """
        logger.debug(query)
        try:
            cursor.execute(
                query, 
                {
                    "query_vector": query_vector,
                    "score_threshold": score_threshold,
                    "k": k
                }
            )
            results = cursor.fetchall()

            chunks = []
            scores = []            
            for row in results:
                content, chunk_id, metadata, chunk_metadata, vector, score = row
                chunk = Chunk(
                    content=content.read(),
                    chunk_id=chunk_id,
                    metadata=metadata,
                    embedding=vector,
                    chunk_metadata=chunk_metadata
                )
                chunks.append(chunk)
                scores.append(float(score))
            return QueryChunksResponse(chunks=chunks, scores=scores)
        except Exception as e:
            logger.error("Error querying vector: %s", e)
            raise
        finally: 
            cursor.close()

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        cursor = self.connection.cursor()
        query = f"""
                SELECT
                    content,
                    chunk_id,
                    metadata,
                    chunk_metadata,
                    vector,
                    score / max_score AS score
                FROM (
                    SELECT
                        content,
                        chunk_id,
                        metadata,
                        chunk_metadata,
                        vector,
                        SCORE(1) AS score,
                        MAX(SCORE(1)) OVER () AS max_score
                    FROM {self.table_name}
                    WHERE CONTAINS(content, :query_string, 1) > 0
                )
                WHERE score >= :score_threshold
                ORDER BY score DESC
                FETCH FIRST :k ROWS ONLY;
                """
        logger.debug(query)

        try:
            cursor.execute(
                query,
                {
                    "query_string": query_string,
                    "score_threshold": score_threshold,
                    "k": k
                },
            )
            results = cursor.fetchall()

            chunks = []
            scores = []
            for row in results:
                content, chunk_id, metadata, chunk_metadata, vector, score = row
                chunk = Chunk(
                    content=content.read(),
                    chunk_id=chunk_id,
                    metadata=metadata,
                    embedding=vector,
                    chunk_metadata=chunk_metadata
                )
                chunks.append(chunk)
                scores.append(float(score))
            return QueryChunksResponse(chunks=chunks, scores=scores)
        except Exception as e:
            logger.error(f"Error performing keyword search: {e}")
            raise
        finally:
            cursor.close()

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        cursor = self.connection.cursor()
        filter_limit = 100  # Should this be set somewhere else?
        query_vector = array("f", embedding.astype("float32"))
        query = f"""
                WITH vec_candidates AS (
                    SELECT content, chunk_id, metadata, chunk_metadata, vector
                    FROM {self.table_name}
                    WHERE 1 - VECTOR_DISTANCE(vector, :query_vector, COSINE) >= :score_threshold
                    ORDER BY VECTOR_DISTANCE(vector, :query_vector, COSINE)
                    FETCH FIRST {filter_limit} ROWS ONLY
                )
                SELECT content, chunk_id, metadata, chunk_metadata, vector,
                    1- VECTOR_DISTANCE(vector, :query_vector, COSINE) AS score
                FROM vec_candidates vc
                WHERE CONTAINS(vc.content, :query_string, 1) > 0
                ORDER BY score
                FETCH FIRST :k ROWS ONLY
            """
        logger.debug(query)

        try:
            cursor.execute(
                query,
                {
                    "query_vector": query_vector,
                    "query_string": query_string,
                    "score_threshold": score_threshold,
                    "k": k
                },
            )
            results = cursor.fetchall()

            chunks = []
            scores = []
            for row in results:
                content, chunk_id, metadata, chunk_metadata, vector, score = row
                chunk = Chunk(
                    content=content.read(),
                    chunk_id=chunk_id,
                    metadata=metadata,
                    embedding=vector,
                    chunk_metadata=chunk_metadata
                )
                chunks.append(chunk)
                scores.append(float(score))
            return QueryChunksResponse(chunks=chunks, scores=scores)
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
        finally:
            cursor.close()

    async def delete(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            logger.info("Dropped table: {self.table_name}")
        except oracledb.DatabaseError as e:
            logger.error(f"Error dropping table {self.table_name}: {e}")
            raise

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE chunk_id IN ({', '.join([f"'{chunk_id}'" for chunk_id in chunk_ids])})
                """
            )
        except Exception as e:
            logger.error(f"Error deleting chunks from Oracle 26AI table {self.table_name}: {e}")
            raise
        finally: 
            cursor.close()


class OCI26aiVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: OCI26aiVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
    ) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        self.config = config
        self.cache = {}
        self.pool = None
        self.inference_api = inference_api
        self.vector_store_table = None

    async def initialize(self) -> None:
        logger.info("Initializing OCI26aiVectorIOAdapter")
        self.kvstore = await kvstore_impl(self.config.persistence)
        await self.initialize_openai_vector_stores()

        try:
            self.connection = oracledb.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=self.config.conn_str,
                config_dir=self.config.tnsnames_loc,
                wallet_location=self.config.ewallet_pem_loc,
                wallet_password=self.config.ewallet_password,
            )
            self.connection.autocommit = True
            logger.info("Oracle connection created successfully")
        except Exception as e:
            logger.error(f"Error creating Oracle connection: {e}")
            raise
        
        # Load State
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            logger.info(f"Loading index {vector_store.vector_store_name}: {vector_store.vector_store_id}")
            oci_index = OCI26aiIndex(
                    connection=self.connection,
                    vector_store=vector_store,
                    kvstore=self.kvstore,
            )
            await oci_index.initialize()
            index = VectorStoreWithIndex(vector_store, index=oci_index, inference_api=self.inference_api)
            self.cache[vector_store.identifier] = index

        logger.info(f"Completed loading {len(stored_vector_stores)} indexes")

    async def shutdown(self) -> None:
        logger.info('Shutting down Oracle connection')
        if self.connection is not None:
            self.connection.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")
        
        # # Save to kvstore for persistence
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())
        
        if isinstance(self.config, OCI26aiVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            consistency_level = "Strong"
        oci_index = OCI26aiIndex(
            connection=self.connection,
                vector_store=vector_store,
                consistency_level=consistency_level,
            )
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=oci_index,
            inference_api=self.inference_api,
        )
        await oci_index.initialize()
        self.cache[vector_store.identifier] = index

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        # Try to load from kvstore
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=OCI26aiIndex(
                connection=self.connection,
                vector_store=vector_store,
                kvstore=self.kvstore,
            ),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        # Remove provider index and cache
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        # Delete vector DB metadata from KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before unregistering vector stores.")
        await self.kvstore.delete(key=f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store_id}")

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
