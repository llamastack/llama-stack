# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
from typing import Any

from numpy.typing import NDArray
import oracledb

from llama_stack.core.storage.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
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
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:oci26ai:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:oci26ai:{VERSION}::"
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
        try:
            with self.connection.cursor() as cursor:
                # Create table
                create_table_sql = f"""
                    CREATE TABLE {self.table_name} (
                        chunk_id VARCHAR2(100) PRIMARY KEY,
                        content CLOB,
                        vector VECTOR({self.dimensions}),
                        chunk_content JSON
                    )
                """
                logger.debug(f"Executing SQL: {create_table_sql}")
                cursor.execute(create_table_sql)
                logger.info(f"Table {self.table_name} created successfully")
        except oracledb.DatabaseError as e:
            if "ORA-00955" in str(e):
                logger.exception(f"Table {self.table_name} already exists")
                raise RuntimeError(f"Error creating index for vector_store: {self.vector_store.identifier}") from e

            else:
                logger.error(f"Error creating table {self.table_name}: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error creating table {self.table_name}: {e}")
            raise

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        data = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "vector": str(embedding.tolist()),  # Convert numpy array to list and then to string
                    "chunk_content": chunk.model_dump(),
                }
            )

        try:
            # TODO: Upsert?
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table_name} (chunk_id, content, vector, chunk_content)
                    VALUES (:chunk_id, :content, VECTOR_FROM_JSON(:vector), :chunk_content)
                    """,
                    data,
                )
            logger.info("Closed connection in add_chunks")
        except Exception as e:
            logger.error(f"Error inserting chunks into Oracle 26AI table {self.table_name}: {e}")
            raise

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        # try:
        #     with self.connection.cursor() as cursor:
        #         cursor.execute(
        #             f"""
        #             SELECT chunk_content, 
        #                 VECTOR_DISTANCE(VECTOR(:embedding), vector, EUCLIDEAN) AS distance
        #             FROM {self.table_name}
        #             ORDER BY distance
        #             FETCH FIRST :k ROWS ONLY
        #             """,
        #             {"embedding": str(embedding.tolist()), "k": k},
        #         )
        #         results = cursor.fetchall()
        #         chunks = [Chunk(**result[0]) for result in results]
        #         scores = [result[1] for result in results]
        #     logger.info("Closed connection in query_vector")
        #     return QueryChunksResponse(chunks=chunks, scores=scores)
        # except Exception as e:
        #     logger.error(f"Error performing vector search: {e}")
        #     raise
        raise NotImplementedError()

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        # try:            
        #     with self.connection.cursor() as cursor:
        #         cursor.execute(
        #             f"""
        #             SELECT chunk_content, 
        #                 1.0 AS score  -- Simple text search doesn't have a score
        #             FROM {self.table_name}
        #             WHERE CONTAINS(content, :query_string) > 0
        #             FETCH FIRST :k ROWS ONLY
        #             """,
        #             {"query_string": query_string, "k": k},
        #         )
        #         results = cursor.fetchall()
        #         chunks = [Chunk(**result[0]) for result in results]
        #         scores = [result[1] for result in results]
        #     return QueryChunksResponse(chunks=chunks, scores=scores)
        # except Exception as e:
        #     logger.error(f"Error performing keyword search: {e}")
        #     raise
        raise NotImplementedError()

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        # try:
        #     with self.connection.cursor() as cursor:
        #         cursor.execute(
        #             f"""
        #             SELECT chunk_content, 
        #                    (VECTOR_DISTANCE(VECTOR(:embedding), vector, EUCLIDEAN) + 
        #                     CASE WHEN CONTAINS(content, :query_string) > 0 THEN 1 ELSE 0 END) AS score
        #             FROM {self.table_name}
        #             ORDER BY score
        #             FETCH FIRST :k ROWS ONLY
        #             """,
        #             {"embedding": str(embedding.tolist()), "query_string": query_string, "k": k},
        #         )
        #         results = cursor.fetchall()
        #         chunks = [Chunk(**result[0]) for result in results]
        #         scores = [result[1] for result in results]
        #         return QueryChunksResponse(chunks=chunks, scores=scores)
        # except Exception as e:
        #     logger.error(f"Error performing hybrid search: {e}")
        #     raise
        raise NotImplementedError()

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
        try:
            with self.connection.cursor() as cursor:    
                cursor.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE chunk_id IN ({', '.join([f"'{chunk_id}'" for chunk_id in chunk_ids])})
                    """
                )
        except Exception as e:
            logger.error(f"Error deleting chunks from Oracle 26AI table {self.table_name}: {e}")
            raise


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

        logger.debug(f"Creating Oracle connection with user: {self.config.user}, dsn: {self.config.conn_str}")
        try:
            self.connection = oracledb.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=self.config.conn_str,
                config_dir=self.config.tnsnames_loc,
                wallet_location=self.config.ewallet_pem_loc,
                wallet_password=self.config.ewallet_password,
            )
            logger.info("Oracle connection created successfully")
        except Exception as e:
            logger.error(f"Error creating Oracle connection: {e}")
            raise
        
        # Load State
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
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
        # if self.kvstore is None:
        #     raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")
        
        # # Save to kvstore for persistence
        # key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        # await self.kvstore.set(key=key, value=vector_store.model_dump_json())
        
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

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
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

# Follow driver installation and setup instructions here: 
# https://www.oracle.com/database/technologies/appdev/python/quickstartpython.html
