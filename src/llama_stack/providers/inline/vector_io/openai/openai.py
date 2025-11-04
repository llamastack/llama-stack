# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.files import Files, OpenAIFileObject
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
    VectorStoreFileDeleteResponse,
    VectorStoreFileLastError,
    VectorStoreFileObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import HealthResponse, HealthStatus, VectorStoresProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion

from .config import OpenAIVectorIOConfig

logger = get_logger(name=__name__, category="vector_io")

# Prefix for storing the mapping from Llama Stack vector store IDs to OpenAI vector store IDs
VECTOR_STORE_ID_MAPPING_PREFIX = "openai_vector_store_id_mapping::"
# Prefix for storing the mapping from Llama Stack file IDs to OpenAI file IDs
FILE_ID_MAPPING_PREFIX = "openai_file_id_mapping::"


class OpenAIVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    def __init__(
        self,
        config: OpenAIVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
    ) -> None:
        super().__init__(files_api=files_api, kvstore=None)
        self.config = config
        self.inference_api = inference_api
        self.openai_client = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)

        # Initialize OpenAI client
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python client library is not installed. Please install it with: pip install openai"
            ) from e

        api_key = self.config.api_key or None
        if api_key == "${OPENAI_API_KEY}":
            api_key = None

        try:
            self.openai_client = AsyncOpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError("Failed to initialize OpenAI client") from e

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def _store_vector_store_id_mapping(self, llama_stack_id: str, openai_id: str) -> None:
        """Store mapping from Llama Stack vector store ID to OpenAI vector store ID."""
        if self.kvstore:
            key = f"{VECTOR_STORE_ID_MAPPING_PREFIX}{llama_stack_id}"
            await self.kvstore.set(key, openai_id)

    async def _get_openai_vector_store_id(self, llama_stack_id: str) -> str:
        """Get OpenAI vector store ID from Llama Stack vector store ID.

        Raises ValueError if mapping is not found.
        """
        if self.kvstore:
            key = f"{VECTOR_STORE_ID_MAPPING_PREFIX}{llama_stack_id}"
            try:
                openai_id = await self.kvstore.get(key)
                if openai_id:
                    return openai_id
            except Exception:
                pass
        # If not found in mapping, raise an error instead of assuming
        raise ValueError(f"No OpenAI vector store mapping found for Llama Stack ID: {llama_stack_id}")

    async def _delete_vector_store_id_mapping(self, llama_stack_id: str) -> None:
        """Delete mapping for a vector store ID."""
        if self.kvstore:
            key = f"{VECTOR_STORE_ID_MAPPING_PREFIX}{llama_stack_id}"
            try:
                await self.kvstore.delete(key)
            except Exception:
                pass

    async def _store_file_id_mapping(self, llama_stack_file_id: str, openai_file_id: str) -> None:
        """Store mapping from Llama Stack file ID to OpenAI file ID."""
        if self.kvstore:
            key = f"{FILE_ID_MAPPING_PREFIX}{llama_stack_file_id}"
            await self.kvstore.set(key, openai_file_id)

    async def _get_openai_file_id(self, llama_stack_file_id: str) -> str | None:
        """Get OpenAI file ID from Llama Stack file ID. Returns None if not found."""
        if self.kvstore:
            key = f"{FILE_ID_MAPPING_PREFIX}{llama_stack_file_id}"
            try:
                openai_id = await self.kvstore.get(key)
                if openai_id:
                    return openai_id
            except Exception:
                pass
        return None

    async def _get_llama_stack_file_id(self, openai_file_id: str) -> str | None:
        """Get Llama Stack file ID from OpenAI file ID. Returns None if not found."""
        if self.kvstore:
            # For reverse lookup, we need to search through all mappings
            prefix = FILE_ID_MAPPING_PREFIX
            start_key = prefix
            end_key = f"{prefix}\xff"
            try:
                items = await self.kvstore.items_in_range(start_key, end_key)
                for key, value in items:
                    if value == openai_file_id:
                        # Extract the Llama Stack file ID from the key
                        return key[len(prefix) :]
            except Exception:
                pass
        return None

    async def _delete_file_id_mapping(self, llama_stack_file_id: str) -> None:
        """Delete mapping for a file ID."""
        if self.kvstore:
            key = f"{FILE_ID_MAPPING_PREFIX}{llama_stack_file_id}"
            try:
                await self.kvstore.delete(key)
            except Exception:
                pass

    async def shutdown(self) -> None:
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to OpenAI API.
        """
        try:
            if self.openai_client is None:
                return HealthResponse(
                    status=HealthStatus.ERROR,
                    message="OpenAI client not initialized",
                )

            # Try to list models as a simple health check
            await self.openai_client.models.list()
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(
                status=HealthStatus.ERROR,
                message=f"Health check failed: {str(e)}",
            )

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """Register a vector store by creating it in OpenAI's API."""
        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized")

        # Create vector store in OpenAI
        created_store = await self.openai_client.vector_stores.create(
            name=vector_store.vector_store_name or vector_store.identifier,
        )

        # Store mapping from Llama Stack ID to OpenAI ID
        await self._store_vector_store_id_mapping(vector_store.identifier, created_store.id)

        logger.info(f"Created OpenAI vector store: {created_store.id} for identifier: {vector_store.identifier}")

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Delete a vector store from OpenAI's API."""
        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Look up the OpenAI ID from our mapping
            if self.kvstore:
                key = f"{VECTOR_STORE_ID_MAPPING_PREFIX}{vector_store_id}"
                try:
                    openai_vector_store_id = await self.kvstore.get(key)
                    if openai_vector_store_id:
                        await self.openai_client.vector_stores.delete(openai_vector_store_id)
                        logger.info(
                            f"Deleted OpenAI vector store: {openai_vector_store_id} for identifier: {vector_store_id}"
                        )
                    else:
                        logger.warning(f"No OpenAI vector store mapping found for {vector_store_id}, skipping deletion")
                except Exception as e:
                    logger.warning(f"Failed to delete vector store {vector_store_id} from OpenAI: {e}", exc_info=True)
            # Clean up the mapping
            await self._delete_vector_store_id_mapping(vector_store_id)
        except Exception as e:
            logger.warning(f"Error in unregister_vector_store for {vector_store_id}: {e}", exc_info=True)

    async def insert_chunks(
        self,
        vector_store_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        OpenAI Vector Stores API doesn't support direct chunk insertion.
        Use file attachment instead via openai_attach_file_to_vector_store.
        """
        raise NotImplementedError(
            "Direct chunk insertion is not supported by OpenAI Vector Stores API. "
            "Please use file attachment instead via the openai_attach_file_to_vector_store endpoint."
        )

    async def query_chunks(
        self,
        vector_store_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """
        OpenAI Vector Stores API doesn't support direct chunk queries.
        Use the OpenAI vector store search API instead.
        """
        raise NotImplementedError(
            "Direct chunk querying is not supported by OpenAI Vector Stores API. "
            "Please use the openai_search_vector_store endpoint instead."
        )

    async def delete_chunks(
        self,
        store_id: str,
        chunks_for_deletion: list[ChunkForDeletion],
    ) -> None:
        """
        OpenAI Vector Stores API doesn't support direct chunk deletion.
        Delete files from the vector store instead.
        """
        raise NotImplementedError(
            "Direct chunk deletion is not supported by OpenAI Vector Stores API. "
            "Please delete files from the vector store instead via openai_delete_vector_store_file."
        )

    async def _prepare_and_attach_file_chunks(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any],
        chunking_strategy: Any,
        created_at: int,
    ) -> tuple[Any, list[Chunk], Any]:
        """
        Override to download file from Llama Stack, upload to OpenAI,
        and attach to OpenAI vector store instead of storing chunks locally.

        Returns: (VectorStoreFileObject, empty chunks list, file response)
        """

        # Translate Llama Stack ID to OpenAI ID
        try:
            openai_vector_store_id = await self._get_openai_vector_store_id(vector_store_id)
        except ValueError as e:
            logger.error(f"Cannot attach file to vector store {vector_store_id}: {e}")
            return (
                VectorStoreFileObject(
                    id=file_id,
                    attributes=attributes,
                    chunking_strategy=chunking_strategy,
                    created_at=created_at,
                    status="failed",
                    vector_store_id=vector_store_id,
                    last_error=VectorStoreFileLastError(
                        code="server_error",
                        message=str(e),
                    ),
                ),
                [],
                None,
            )

        vector_store_file_object = VectorStoreFileObject(
            id=file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
            created_at=created_at,
            status="in_progress",
            vector_store_id=vector_store_id,
        )

        # Prepare file: download from Llama Stack if needed, upload to OpenAI
        try:
            file_obj: OpenAIFileObject = await self.files_api.openai_retrieve_file(file_id)
            file_content_response = await self.files_api.openai_retrieve_file_content(file_id)
            file_data = file_content_response.body

            import io

            file_buffer = io.BytesIO(file_data)
            file_buffer.name = file_obj.filename

            openai_file = await self.openai_client.files.create(
                file=file_buffer,
                purpose="assistants",
            )

            logger.info(f"Uploaded file {file_id} to OpenAI as {openai_file.id}")
            openai_file_id = openai_file.id
            # Store mapping for later lookup
            await self._store_file_id_mapping(file_id, openai_file_id)
        except Exception as e:
            logger.debug(f"Could not retrieve file {file_id} from Llama Stack: {e}. Using file_id directly.")
            openai_file_id = file_id

        # Attach file to OpenAI vector store
        try:
            attached_file = await self.openai_client.vector_stores.files.create(
                vector_store_id=openai_vector_store_id,
                file_id=openai_file_id,
            )

            logger.info(
                f"Attached file {openai_file_id} to OpenAI vector store {openai_vector_store_id}, "
                f"status: {attached_file.status}"
            )

            # Use the status from OpenAI's response, don't assume it's completed
            vector_store_file_object.status = attached_file.status
            file_response = file_obj if "file_obj" in locals() else None
        except Exception as e:
            logger.error(f"Failed to attach file {openai_file_id} to vector store: {e}")
            vector_store_file_object.status = "failed"
            vector_store_file_object.last_error = VectorStoreFileLastError(
                code="server_error",
                message=str(e),
            )
            file_response = file_obj if "file_obj" in locals() else None

        # Return VectorStoreFileObject and empty chunks (OpenAI handles storage)
        return vector_store_file_object, [], file_response

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
        """Search a vector store using OpenAI's native search API."""
        assert self.openai_client is not None

        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        openai_vector_store_id = await self._get_openai_vector_store_id(vector_store_id)
        # raise ValueError(f"openai_vector_store_id: {openai_vector_store_id}")
        logger.info(f"openai_vector_store_id: {openai_vector_store_id}")
        response = await self.openai_client.vector_stores.search(
            vector_store_id=openai_vector_store_id,
            query=query,
            filters=filters,
            max_num_results=max_num_results,
            ranking_options=ranking_options,
            rewrite_query=rewrite_query,
        )
        payload = response.model_dump()
        logger.info(f"payload: {payload}")
        # Remap OpenAI file IDs back to Llama Stack file IDs in results
        if payload.get("data"):
            for result in payload["data"]:
                if result.get("file_id"):
                    llama_stack_file_id = await self._get_llama_stack_file_id(result["file_id"])
                    if llama_stack_file_id:
                        result["file_id"] = llama_stack_file_id

        return VectorStoreSearchResponsePage(**payload)

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        """Delete a file from a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Get the OpenAI file ID for this Llama Stack file ID
            openai_file_id = await self._get_openai_file_id(file_id)
            if not openai_file_id:
                # If no mapping, use the file_id as-is (may be native OpenAI file ID)
                openai_file_id = file_id

            # Get OpenAI vector store ID
            openai_vector_store_id = await self._get_openai_vector_store_id(vector_store_id)

            # Delete file from OpenAI vector store
            await self.openai_client.vector_stores.files.delete(
                vector_store_id=openai_vector_store_id,
                file_id=openai_file_id,
            )

            logger.info(f"Deleted file {openai_file_id} from OpenAI vector store {openai_vector_store_id}")

            # Delete the file from OpenAI if it was created by us
            if await self._get_openai_file_id(file_id):
                try:
                    await self.openai_client.files.delete(openai_file_id)
                    logger.info(f"Deleted OpenAI file {openai_file_id}")
                except Exception as e:
                    logger.debug(f"Could not delete OpenAI file {openai_file_id}: {e}")

            # Clean up mappings
            await self._delete_file_id_mapping(file_id)

            # Update vector store metadata
            store_info = self.openai_vector_stores[vector_store_id].copy()
            if file_id in store_info["file_ids"]:
                store_info["file_ids"].remove(file_id)
                store_info["file_counts"]["total"] -= 1
                store_info["file_counts"]["completed"] -= 1
                self.openai_vector_stores[vector_store_id] = store_info
                await self._save_openai_vector_store(vector_store_id, store_info)

            return VectorStoreFileDeleteResponse(
                id=file_id,
                deleted=True,
            )

        except Exception as e:
            logger.error(f"Error deleting file {file_id} from vector store {vector_store_id}: {e}")
            raise

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        """Retrieve a vector store file and check status from OpenAI if still in_progress."""
        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized")

        # Get the cached file info
        file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
        file_object = VectorStoreFileObject(**file_info)

        # If status is still in_progress, check the actual status from OpenAI
        if file_object.status == "in_progress":
            try:
                # Get OpenAI file ID for this Llama Stack file ID
                openai_file_id = await self._get_openai_file_id(file_id)
                if not openai_file_id:
                    openai_file_id = file_id

                # Get OpenAI vector store ID
                openai_vector_store_id = await self._get_openai_vector_store_id(vector_store_id)

                # Retrieve the file status from OpenAI
                openai_file = await self.openai_client.vector_stores.files.retrieve(
                    vector_store_id=openai_vector_store_id,
                    file_id=openai_file_id,
                )

                # Update the status from OpenAI
                file_object.status = openai_file.status

                # If status changed, update it in storage
                if openai_file.status != "in_progress":
                    file_info["status"] = openai_file.status
                    # Update file counts in vector store metadata
                    store_info = self.openai_vector_stores[vector_store_id].copy()
                    if file_object.status == "completed":
                        store_info["file_counts"]["in_progress"] = max(
                            0, store_info["file_counts"].get("in_progress", 0) - 1
                        )
                        store_info["file_counts"]["completed"] = (
                            store_info["file_counts"].get("completed", 0) + 1
                        )
                    elif file_object.status == "failed":
                        store_info["file_counts"]["in_progress"] = max(
                            0, store_info["file_counts"].get("in_progress", 0) - 1
                        )
                        store_info["file_counts"]["failed"] = store_info["file_counts"].get("failed", 0) + 1

                    self.openai_vector_stores[vector_store_id] = store_info
                    await self._save_openai_vector_store_file(vector_store_id, file_id, file_info)
                    await self._save_openai_vector_store(vector_store_id, store_info)

            except Exception as e:
                logger.debug(f"Could not retrieve file status from OpenAI: {e}. Using cached status.")

        return file_object
