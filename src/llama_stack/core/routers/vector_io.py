# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import Body

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.models import ModelType
from llama_stack.apis.vector_io import (
    Chunk,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import HealthResponse, HealthStatus, RoutingTable

logger = get_logger(name=__name__, category="core::routers")


class VectorIORouter(VectorIO):
    """Routes to an provider based on the vector db identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
        vector_stores_config: VectorStoresConfig | None = None,
    ) -> None:
        logger.debug("Initializing VectorIORouter")
        self.routing_table = routing_table
        self.vector_stores_config = vector_stores_config

    async def initialize(self) -> None:
        logger.debug("VectorIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("VectorIORouter.shutdown")
        pass

    async def _get_embedding_model_dimension(self, embedding_model_id: str) -> int:
        """Get the embedding dimension for a specific embedding model."""
        all_models = await self.routing_table.get_all_with_type("model")

        for model in all_models:
            if model.identifier == embedding_model_id and model.model_type == ModelType.embedding:
                dimension = model.metadata.get("embedding_dimension")
                if dimension is None:
                    raise ValueError(f"Embedding model '{embedding_model_id}' has no embedding_dimension in metadata")
                return int(dimension)

        raise ValueError(f"Embedding model '{embedding_model_id}' not found or not an embedding model")

    async def insert_chunks(
        self,
        vector_store_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        doc_ids = [chunk.document_id for chunk in chunks[:3]]
        logger.debug(
            f"VectorIORouter.insert_chunks: {vector_store_id}, {len(chunks)} chunks, "
            f"ttl_seconds={ttl_seconds}, chunk_ids={doc_ids}{' and more...' if len(chunks) > 3 else ''}"
        )
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.insert_chunks(vector_store_id, chunks, ttl_seconds)

    async def query_chunks(
        self,
        vector_store_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        logger.debug(f"VectorIORouter.query_chunks: {vector_store_id}")
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.query_chunks(vector_store_id, query, params)

    # OpenAI Vector Stores API endpoints
    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        # Extract llama-stack-specific parameters from extra_body
        extra = params.model_extra or {}
        embedding_model = extra.get("embedding_model")
        embedding_dimension = extra.get("embedding_dimension")
        provider_id = extra.get("provider_id")

        # Use default embedding model if not specified
        if (
            embedding_model is None
            and self.vector_stores_config
            and self.vector_stores_config.default_embedding_model is not None
        ):
            # Construct the full model ID with provider prefix
            embedding_provider_id = self.vector_stores_config.default_embedding_model.provider_id
            model_id = self.vector_stores_config.default_embedding_model.model_id
            embedding_model = f"{embedding_provider_id}/{model_id}"

        if embedding_model is not None and embedding_dimension is None:
            embedding_dimension = await self._get_embedding_model_dimension(embedding_model)

        # Auto-select provider if not specified
        if provider_id is None:
            num_providers = len(self.routing_table.impls_by_provider_id)
            if num_providers == 0:
                raise ValueError("No vector_io providers available")
            if num_providers > 1:
                available_providers = list(self.routing_table.impls_by_provider_id.keys())
                # Use default configured provider
                if self.vector_stores_config and self.vector_stores_config.default_provider_id:
                    default_provider = self.vector_stores_config.default_provider_id
                    if default_provider in available_providers:
                        provider_id = default_provider
                        logger.debug(f"Using configured default vector store provider: {provider_id}")
                    else:
                        raise ValueError(
                            f"Configured default vector store provider '{default_provider}' not found. "
                            f"Available providers: {available_providers}"
                        )
                else:
                    raise ValueError(
                        f"Multiple vector_io providers available. Please specify provider_id in extra_body. "
                        f"Available providers: {available_providers}"
                    )
            else:
                provider_id = list(self.routing_table.impls_by_provider_id.keys())[0]

        vector_store_id = f"vs_{uuid.uuid4()}"
        registered_vector_store = await self.routing_table.register_vector_store(
            vector_store_id=vector_store_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id=provider_id,
            provider_vector_store_id=vector_store_id,
            vector_store_name=params.name,
        )
        provider = await self.routing_table.get_provider_impl(registered_vector_store.identifier)

        # Update model_extra with registered values so provider uses the already-registered vector_store
        if params.model_extra is None:
            params.model_extra = {}
        params.model_extra["provider_vector_store_id"] = registered_vector_store.provider_resource_id
        params.model_extra["provider_id"] = registered_vector_store.provider_id
        if embedding_model is not None:
            params.model_extra["embedding_model"] = embedding_model
        if embedding_dimension is not None:
            params.model_extra["embedding_dimension"] = embedding_dimension

        # Set chunking strategy explicitly if not provided
        if params.chunking_strategy is None or params.chunking_strategy.type == "auto":
            # actualize the chunking strategy to static
            params.chunking_strategy = VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig()
            )

        return await provider.openai_create_vector_store(params)

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
        vector_stores = await self.routing_table.get_all_with_type("vector_store")
        all_stores = []
        for vector_store in vector_stores:
            try:
                provider = await self.routing_table.get_provider_impl(vector_store.identifier)
                vector_store = await provider.openai_retrieve_vector_store(vector_store.identifier)
                all_stores.append(vector_store)
            except Exception as e:
                logger.error(f"Error retrieving vector store {vector_store.identifier}: {e}")
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
            before_index = next(
                (i for i, store in enumerate(all_stores) if store.id == before),
                len(all_stores),
            )
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
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store(vector_store_id)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        logger.debug(f"VectorIORouter.openai_update_vector_store: {vector_store_id}")
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_update_vector_store(
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
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_search_vector_store(
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
        if chunking_strategy is None or chunking_strategy.type == "auto":
            chunking_strategy = VectorStoreChunkingStrategyStatic(static=VectorStoreChunkingStrategyStaticConfig())
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_attach_file_to_vector_store(
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
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_list_files_in_vector_store(
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
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileContentResponse:
        logger.debug(f"VectorIORouter.openai_retrieve_vector_store_file_contents: {vector_store_id}, {file_id}")
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file_contents(
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
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_update_vector_store_file(
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
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_delete_vector_store_file(
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

    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        logger.debug(
            f"VectorIORouter.openai_create_vector_store_file_batch: {vector_store_id}, {len(params.file_ids)} files"
        )
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_create_vector_store_file_batch(vector_store_id, params)

    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        logger.debug(f"VectorIORouter.openai_retrieve_vector_store_file_batch: {batch_id}, {vector_store_id}")
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
        )

    async def openai_list_files_in_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
        after: str | None = None,
        before: str | None = None,
        filter: str | None = None,
        limit: int | None = 20,
        order: str | None = "desc",
    ) -> VectorStoreFilesListInBatchResponse:
        logger.debug(f"VectorIORouter.openai_list_files_in_vector_store_file_batch: {batch_id}, {vector_store_id}")
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_list_files_in_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
            after=after,
            before=before,
            filter=filter,
            limit=limit,
            order=order,
        )

    async def openai_cancel_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        logger.debug(f"VectorIORouter.openai_cancel_vector_store_file_batch: {batch_id}, {vector_store_id}")
        provider = await self.routing_table.get_provider_impl(vector_store_id)
        return await provider.openai_cancel_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
        )
