# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import TypeAdapter

from llama_stack.apis.common.errors import ModelNotFoundError, ModelTypeError, VectorStoreNotFoundError
from llama_stack.apis.models import ModelType
from llama_stack.apis.resource import ResourceType
from llama_stack.apis.vector_dbs import ListVectorDBsResponse, VectorDB, VectorDBs
from llama_stack.apis.vector_io.vector_io import (
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreDeleteResponse,
    VectorStoreFileContentsResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFileStatus,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.core.datatypes import (
    VectorDBWithOwner,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl, lookup_model

logger = get_logger(name=__name__, category="core::routing_tables")


class VectorDBsRoutingTable(CommonRoutingTableImpl, VectorDBs):
    async def list_vector_dbs(self) -> ListVectorDBsResponse:
        return ListVectorDBsResponse(data=await self.get_all_with_type("vector_db"))

    async def get_vector_db(self, vector_db_id: str) -> VectorDB:
        vector_db = await self.get_object_by_identifier("vector_db", vector_db_id)
        if vector_db is None:
            raise VectorStoreNotFoundError(vector_db_id)
        return vector_db

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
        vector_db_name: str | None = None,
    ) -> VectorDB:
        if provider_id is None:
            if len(self.impls_by_provider_id) > 0:
                provider_id = list(self.impls_by_provider_id.keys())[0]
                if len(self.impls_by_provider_id) > 1:
                    logger.warning(
                        f"No provider specified and multiple providers available. Arbitrarily selected the first provider {provider_id}."
                    )
            else:
                raise ValueError("No provider available. Please configure a vector_io provider.")
        model = await lookup_model(self, embedding_model)
        if model is None:
            raise ModelNotFoundError(embedding_model)
        if model.model_type != ModelType.embedding:
            raise ModelTypeError(embedding_model, model.model_type, ModelType.embedding)
        if "embedding_dimension" not in model.metadata:
            raise ValueError(f"Model {embedding_model} does not have an embedding dimension")

        provider = self.impls_by_provider_id[provider_id]
        logger.warning(
            "VectorDB is being deprecated in future releases in favor of VectorStore. Please migrate your usage accordingly."
        )
        vector_store = await provider.openai_create_vector_store(
            name=vector_db_name or vector_db_id,
            embedding_model=embedding_model,
            embedding_dimension=model.metadata["embedding_dimension"],
            provider_id=provider_id,
            provider_vector_db_id=provider_vector_db_id,
        )

        vector_store_id = vector_store.id
        actual_provider_vector_db_id = provider_vector_db_id or vector_store_id
        logger.warning(
            f"Ignoring vector_db_id {vector_db_id} and using vector_store_id {vector_store_id} instead. Setting VectorDB {vector_db_id} to VectorDB.vector_db_name"
        )

        vector_db_data = {
            "identifier": vector_store_id,
            "type": ResourceType.vector_db.value,
            "provider_id": provider_id,
            "provider_resource_id": actual_provider_vector_db_id,
            "embedding_model": embedding_model,
            "embedding_dimension": model.metadata["embedding_dimension"],
            "vector_db_name": vector_store.name,
        }
        vector_db = TypeAdapter(VectorDBWithOwner).validate_python(vector_db_data)
        await self.register_object(vector_db)
        return vector_db

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        existing_vector_db = await self.get_vector_db(vector_db_id)
        await self.unregister_object(existing_vector_db)

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        await self.assert_action_allowed("read", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store(vector_store_id)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        await self.assert_action_allowed("update", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
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
        await self.assert_action_allowed("delete", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        result = await provider.openai_delete_vector_store(vector_store_id)
        await self.unregister_vector_db(vector_store_id)
        return result

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
        await self.assert_action_allowed("read", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
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
        await self.assert_action_allowed("update", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
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
        await self.assert_action_allowed("read", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
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
        await self.assert_action_allowed("read", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileContentsResponse:
        await self.assert_action_allowed("read", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
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
        await self.assert_action_allowed("update", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
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
        await self.assert_action_allowed("delete", "vector_db", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_delete_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
