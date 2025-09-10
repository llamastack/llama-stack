# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import uuid
from typing import Any

from llama_stack.apis.common.content_types import (
    InterleavedContent,
)
from llama_stack.apis.models import ModelType
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
    VectorStoreChunkingStrategy,
    VectorStoreDeleteResponse,
    VectorStoreFileContentsResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFileStatus,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import HealthResponse, HealthStatus, RoutingTable

logger = get_logger(name=__name__, category="core::routers")


class VectorIORouter(VectorIO):
    """Routes to an provider based on the vector db identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing VectorIORouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("VectorIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("VectorIORouter.shutdown")
        pass

    async def _get_first_embedding_model(self) -> tuple[str, int] | None:
        """Get the first available embedding model identifier."""
        try:
            # Get all models from the routing table
            all_models = await self.routing_table.get_all_with_type("model")

            # Filter for embedding models
            embedding_models = [
                model
                for model in all_models
                if hasattr(model, "model_type") and model.model_type == ModelType.embedding
            ]

            if embedding_models:
                dimension = embedding_models[0].metadata.get("embedding_dimension", None)
                if dimension is None:
                    raise ValueError(f"Embedding model {embedding_models[0].identifier} has no embedding dimension")
                return embedding_models[0].identifier, dimension
            else:
                logger.warning("No embedding models found in the routing table")
                return None
        except Exception as e:
            logger.error(f"Error getting embedding models: {e}")
            return None

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        vector_db_name: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> None:
        logger.debug(f"VectorIORouter.register_vector_db: {vector_db_id}, {embedding_model}")
        await self.routing_table.register_vector_db(
            vector_db_id,
            embedding_model,
            embedding_dimension,
            provider_id,
            vector_db_name,
            provider_vector_db_id,
        )

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        logger.debug(
            f"VectorIORouter.insert_chunks: {vector_db_id}, {len(chunks)} chunks, ttl_seconds={ttl_seconds}, chunk_ids={[chunk.metadata['document_id'] for chunk in chunks[:3]]}{' and more...' if len(chunks) > 3 else ''}",
        )
        provider = await self.routing_table.get_provider_impl(vector_db_id)
        return await provider.insert_chunks(vector_db_id, chunks, ttl_seconds)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        logger.debug(f"VectorIORouter.query_chunks: {vector_db_id}")
        provider = await self.routing_table.get_provider_impl(vector_db_id)
        return await provider.query_chunks(vector_db_id, query, params)

    # OpenAI Vector Stores API endpoints
    async def openai_create_vector_store(
        self,
        name: str,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        provider_id: str | None = None,
    ) -> VectorStoreObject:
        logger.debug(f"VectorIORouter.openai_create_vector_store: name={name}, provider_id={provider_id}")

        # If no embedding model is provided, use the first available one
        if embedding_model is None:
            embedding_model_info = await self._get_first_embedding_model()
            if embedding_model_info is None:
                raise ValueError("No embedding model provided and no embedding models available in the system")
            embedding_model, embedding_dimension = embedding_model_info
            logger.info(f"No embedding model specified, using first available: {embedding_model}")

        vector_db_id = f"vs_{uuid.uuid4()}"
        registered_vector_db = await self.routing_table.register_vector_db(
            vector_db_id=vector_db_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id=provider_id,
            provider_vector_db_id=vector_db_id,
            vector_db_name=name,
        )
        provider = await self.routing_table.get_provider_impl(registered_vector_db.identifier)
        return await provider.openai_create_vector_store(
            name=name,
            file_ids=file_ids,
            expires_after=expires_after,
            chunking_strategy=chunking_strategy,
            metadata=metadata,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id=registered_vector_db.provider_id,
            provider_vector_db_id=registered_vector_db.provider_resource_id,
        )

    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        logger.debug(f"VectorIORouter.openai_list_vector_stores: limit={limit}")
        # Route to default provider for now - could aggregate from all providers in the future
        # call retrieve on each vector dbs to get list of vector stores
        vector_dbs = await self.routing_table.get_all_with_type("vector_db")
        all_stores = []
        for vector_db in vector_dbs:
            try:
                provider = await self.routing_table.get_provider_impl(vector_db.identifier)
                vector_store = await provider.openai_retrieve_vector_store(vector_db.identifier)
                all_stores.append(vector_store)
            except Exception as e:
                logger.error(f"Error retrieving vector store {vector_db.identifier}: {e}")
                continue

        # Sort by created_at
        reverse_order = order == "desc"
        all_stores.sort(key=lambda x: x.created_at, reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, store in enumerate(all_stores) if store.id == after), -1)
            if after_index >= 0:
                all_stores = all_stores[after_index + 1 :]

        if before:
            before_index = next((i for i, store in enumerate(all_stores) if store.id == before), len(all_stores))
            all_stores = all_stores[:before_index]

        # Apply limit
        limited_stores = all_stores[:limit]

        # Determine pagination info
        has_more = len(all_stores) > limit
        first_id = limited_stores[0].id if limited_stores else None
        last_id = limited_stores[-1].id if limited_stores else None

        return VectorStoreListResponse(
            data=limited_stores,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        logger.debug(f"VectorIORouter.openai_retrieve_vector_store: {vector_store_id}")
        return await self.routing_table.openai_retrieve_vector_store(vector_store_id)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        logger.debug(f"VectorIORouter.openai_update_vector_store: {vector_store_id}")
        return await self.routing_table.openai_update_vector_store(
            vector_store_id=vector_store_id,
            name=name,
            expires_after=expires_after,
            metadata=metadata,
        )

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        logger.debug(f"VectorIORouter.openai_delete_vector_store: {vector_store_id}")
        return await self.routing_table.openai_delete_vector_store(vector_store_id)

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = 10,
        ranking_options: SearchRankingOptions | None = None,
        rewrite_query: bool | None = False,
        search_mode: str | None = "vector",
    ) -> VectorStoreSearchResponsePage:
        logger.debug(f"VectorIORouter.openai_search_vector_store: {vector_store_id}")
        return await self.routing_table.openai_search_vector_store(
            vector_store_id=vector_store_id,
            query=query,
            filters=filters,
            max_num_results=max_num_results,
            ranking_options=ranking_options,
            rewrite_query=rewrite_query,
            search_mode=search_mode,
        )

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> VectorStoreFileObject:
        logger.debug(f"VectorIORouter.openai_attach_file_to_vector_store: {vector_store_id}, {file_id}")
        return await self.routing_table.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
        )

    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> list[VectorStoreFileObject]:
        logger.debug(f"VectorIORouter.openai_list_files_in_vector_store: {vector_store_id}")
        return await self.routing_table.openai_list_files_in_vector_store(
            vector_store_id=vector_store_id,
            limit=limit,
            order=order,
            after=after,
            before=before,
            filter=filter,
        )

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        logger.debug(f"VectorIORouter.openai_retrieve_vector_store_file: {vector_store_id}, {file_id}")
        return await self.routing_table.openai_retrieve_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileContentsResponse:
        logger.debug(f"VectorIORouter.openai_retrieve_vector_store_file_contents: {vector_store_id}, {file_id}")
        return await self.routing_table.openai_retrieve_vector_store_file_contents(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any],
    ) -> VectorStoreFileObject:
        logger.debug(f"VectorIORouter.openai_update_vector_store_file: {vector_store_id}, {file_id}")
        return await self.routing_table.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes,
        )

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        logger.debug(f"VectorIORouter.openai_delete_vector_store_file: {vector_store_id}, {file_id}")
        return await self.routing_table.openai_delete_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def health(self) -> dict[str, HealthResponse]:
        health_statuses = {}
        timeout = 1  # increasing the timeout to 1 second for health checks
        for provider_id, impl in self.routing_table.impls_by_provider_id.items():
            try:
                # check if the provider has a health method
                if not hasattr(impl, "health"):
                    continue
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                health_statuses[provider_id] = health
            except TimeoutError:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR,
                    message=f"Health check timed out after {timeout} seconds",
                )
            except NotImplementedError:
                health_statuses[provider_id] = HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
            except Exception as e:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"
                )
        return health_statuses
