# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import mimetypes
import time
import uuid
from abc import ABC, abstractmethod
from typing import Annotated, Any

from fastapi import Body
from pydantic import TypeAdapter

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files import Files, OpenAIFileObject
from llama_stack.apis.vector_io import (
    Chunk,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreContent,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileCounts,
    VectorStoreFileDeleteResponse,
    VectorStoreFileLastError,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.core.id_generation import generate_object_id
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    content_from_data_and_mime_type,
    make_overlapped_chunks,
)

EMBEDDING_DIMENSION = 768

logger = get_logger(name=__name__, category="providers::utils")

# Constants for OpenAI vector stores
CHUNK_MULTIPLIER = 5
FILE_BATCH_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60  # 1 day in seconds
MAX_CONCURRENT_FILES_PER_BATCH = 3  # Maximum concurrent file processing within a batch
FILE_BATCH_CHUNK_SIZE = 10  # Process files in chunks of this size

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:{VERSION}::"
OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX = f"openai_vector_stores_file_batches:{VERSION}::"


class OpenAIVectorStoreMixin(ABC):
    """
    Mixin class that provides common OpenAI Vector Store API implementation.
    Providers need to implement the abstract storage methods and maintain
    an openai_vector_stores in-memory cache.
    """

    # Implementing classes should call super().__init__() in their __init__ method
    # to properly initialize the mixin attributes.
    def __init__(
        self,
        files_api: Files | None = None,
        kvstore: KVStore | None = None,
    ):
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}
        self.openai_file_batches: dict[str, dict[str, Any]] = {}
        self.files_api = files_api
        self.kvstore = kvstore
        self._last_file_batch_cleanup_time = 0
        self._file_batch_tasks: dict[str, asyncio.Task[None]] = {}

    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Save vector store metadata to persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))
        # update in-memory cache
        self.openai_vector_stores[store_id] = store_info

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load all vector store metadata from persistent storage."""
        assert self.kvstore
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored_data = await self.kvstore.values_in_range(start_key, end_key)

        stores: dict[str, dict[str, Any]] = {}
        for item in stored_data:
            info = json.loads(item)
            stores[info["id"]] = info
        return stores

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update vector store metadata in persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))
        # update in-memory cache
        self.openai_vector_stores[store_id] = store_info

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete vector store metadata from persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.delete(key)
        # remove from in-memory cache
        self.openai_vector_stores.pop(store_id, None)

    async def _save_openai_vector_store_file(
        self,
        store_id: str,
        file_id: str,
        file_info: dict[str, Any],
        file_contents: list[dict[str, Any]],
    ) -> None:
        """Save vector store file metadata to persistent storage."""
        assert self.kvstore
        meta_key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        await self.kvstore.set(key=meta_key, value=json.dumps(file_info))
        contents_prefix = f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}{store_id}:{file_id}:"
        for idx, chunk in enumerate(file_contents):
            await self.kvstore.set(key=f"{contents_prefix}{idx}", value=json.dumps(chunk))

    async def _load_openai_vector_store_file(self, store_id: str, file_id: str) -> dict[str, Any]:
        """Load vector store file metadata from persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        stored_data = await self.kvstore.get(key)
        return json.loads(stored_data) if stored_data else {}

    async def _load_openai_vector_store_file_contents(self, store_id: str, file_id: str) -> list[dict[str, Any]]:
        """Load vector store file contents from persistent storage."""
        assert self.kvstore
        prefix = f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}{store_id}:{file_id}:"
        end_key = f"{prefix}\xff"
        raw_items = await self.kvstore.values_in_range(prefix, end_key)
        return [json.loads(item) for item in raw_items]

    async def _update_openai_vector_store_file(self, store_id: str, file_id: str, file_info: dict[str, Any]) -> None:
        """Update vector store file metadata in persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        await self.kvstore.set(key=key, value=json.dumps(file_info))

    async def _delete_openai_vector_store_file_from_storage(self, store_id: str, file_id: str) -> None:
        """Delete vector store file metadata from persistent storage."""
        assert self.kvstore

        meta_key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        await self.kvstore.delete(meta_key)

        contents_prefix = f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}{store_id}:{file_id}:"
        end_key = f"{contents_prefix}\xff"
        # load all stored chunk values (values_in_range is implemented by all backends)
        raw_items = await self.kvstore.values_in_range(contents_prefix, end_key)
        # delete each chunk by its index suffix
        for idx in range(len(raw_items)):
            await self.kvstore.delete(f"{contents_prefix}{idx}")

    async def _save_openai_vector_store_file_batch(self, batch_id: str, batch_info: dict[str, Any]) -> None:
        """Save file batch metadata to persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch_id}"
        await self.kvstore.set(key=key, value=json.dumps(batch_info))
        # update in-memory cache
        self.openai_file_batches[batch_id] = batch_info

    async def _load_openai_vector_store_file_batches(self) -> dict[str, dict[str, Any]]:
        """Load all file batch metadata from persistent storage."""
        assert self.kvstore
        start_key = OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}\xff"
        stored_data = await self.kvstore.values_in_range(start_key, end_key)

        batches: dict[str, dict[str, Any]] = {}
        for item in stored_data:
            info = json.loads(item)
            batches[info["id"]] = info
        return batches

    async def _delete_openai_vector_store_file_batch(self, batch_id: str) -> None:
        """Delete file batch metadata from persistent storage and in-memory cache."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch_id}"
        await self.kvstore.delete(key)
        # remove from in-memory cache
        self.openai_file_batches.pop(batch_id, None)

    async def _cleanup_expired_file_batches(self) -> None:
        """Clean up expired file batches from persistent storage."""
        assert self.kvstore
        start_key = OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}\xff"
        stored_data = await self.kvstore.values_in_range(start_key, end_key)

        current_time = int(time.time())
        expired_count = 0

        for item in stored_data:
            info = json.loads(item)
            expires_at = info.get("expires_at")
            if expires_at and current_time > expires_at:
                logger.info(f"Cleaning up expired file batch: {info['id']}")
                await self.kvstore.delete(f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{info['id']}")
                # Remove from in-memory cache if present
                self.openai_file_batches.pop(info["id"], None)
                expired_count += 1

        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired file batches")

    async def _get_completed_files_in_batch(self, vector_store_id: str, file_ids: list[str]) -> set[str]:
        """Determine which files in a batch are actually completed by checking vector store file_ids."""
        if vector_store_id not in self.openai_vector_stores:
            return set()

        store_info = self.openai_vector_stores[vector_store_id]
        completed_files = set(file_ids) & set(store_info["file_ids"])
        return completed_files

    async def _analyze_batch_completion_on_resume(self, batch_id: str, batch_info: dict[str, Any]) -> list[str]:
        """Analyze batch completion status and return remaining files to process.

        Returns:
            List of file IDs that still need processing. Empty list if batch is complete.
        """
        vector_store_id = batch_info["vector_store_id"]
        all_file_ids = batch_info["file_ids"]

        # Find files that are actually completed
        completed_files = await self._get_completed_files_in_batch(vector_store_id, all_file_ids)
        remaining_files = [file_id for file_id in all_file_ids if file_id not in completed_files]

        completed_count = len(completed_files)
        total_count = len(all_file_ids)
        remaining_count = len(remaining_files)

        # Update file counts to reflect actual state
        batch_info["file_counts"] = {
            "completed": completed_count,
            "failed": 0,  # We don't track failed files during resume - they'll be retried
            "in_progress": remaining_count,
            "cancelled": 0,
            "total": total_count,
        }

        # If all files are already completed, mark batch as completed
        if remaining_count == 0:
            batch_info["status"] = "completed"
            logger.info(f"Batch {batch_id} is already fully completed, updating status")

        # Save updated batch info
        await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        return remaining_files

    async def _resume_incomplete_batches(self) -> None:
        """Resume processing of incomplete file batches after server restart."""
        for batch_id, batch_info in self.openai_file_batches.items():
            if batch_info["status"] == "in_progress":
                logger.info(f"Analyzing incomplete file batch: {batch_id}")

                remaining_files = await self._analyze_batch_completion_on_resume(batch_id, batch_info)

                # Check if batch is now completed after analysis
                if batch_info["status"] == "completed":
                    continue

                if remaining_files:
                    logger.info(f"Resuming batch {batch_id} with {len(remaining_files)} remaining files")
                    # Restart the background processing task with only remaining files
                    task = asyncio.create_task(self._process_file_batch_async(batch_id, batch_info, remaining_files))
                    self._file_batch_tasks[batch_id] = task

    async def initialize_openai_vector_stores(self) -> None:
        """Load existing OpenAI vector stores and file batches into the in-memory cache."""
        self.openai_vector_stores = await self._load_openai_vector_stores()
        self.openai_file_batches = await self._load_openai_vector_store_file_batches()
        self._file_batch_tasks = {}
        # TODO: Resume only works for single worker deployment. Jobs with multiple workers will need to be handled differently.
        await self._resume_incomplete_batches()
        self._last_file_batch_cleanup_time = 0

    async def shutdown(self) -> None:
        """Clean up mixin resources including background tasks."""
        # Cancel any running file batch tasks gracefully
        tasks_to_cancel = list(self._file_batch_tasks.items())
        for _, task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @abstractmethod
    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from a vector store."""
        pass

    @abstractmethod
    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """Register a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Unregister a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def insert_chunks(
        self,
        vector_store_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert chunks into a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def query_chunks(
        self, vector_store_id: str, query: Any, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        """Query chunks from a vector database (provider-specific implementation)."""
        pass

    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        """Creates a vector store."""
        created_at = int(time.time())

        # Extract llama-stack-specific parameters from extra_body
        extra_body = params.model_extra or {}
        metadata = params.metadata or {}

        provider_vector_store_id = extra_body.get("provider_vector_store_id")

        # Use embedding info from metadata if available, otherwise from extra_body
        if metadata.get("embedding_model"):
            # If either is in metadata, use metadata as source
            embedding_model = metadata.get("embedding_model")
            embedding_dimension = (
                int(metadata["embedding_dimension"]) if metadata.get("embedding_dimension") else EMBEDDING_DIMENSION
            )
            logger.debug(
                f"Using embedding config from metadata (takes precedence over extra_body): model='{embedding_model}', dimension={embedding_dimension}"
            )
        else:
            embedding_model = extra_body.get("embedding_model")
            embedding_dimension = extra_body.get("embedding_dimension", EMBEDDING_DIMENSION)
            logger.debug(
                f"Using embedding config from extra_body: model='{embedding_model}', dimension={embedding_dimension}"
            )

        # use provider_id set by router; fallback to provider's own ID when used directly via --stack-config
        provider_id = extra_body.get("provider_id") or getattr(self, "__provider_id__", None)
        # Derive the canonical vector_store_id (allow override, else generate)
        vector_store_id = provider_vector_store_id or generate_object_id("vector_store", lambda: f"vs_{uuid.uuid4()}")

        if embedding_model is None:
            raise ValueError("embedding_model is required")

        if embedding_dimension is None:
            raise ValueError("Embedding dimension is required")

        # Register the VectorStore backing this vector store
        if provider_id is None:
            raise ValueError("Provider ID is required but was not provided")

        # call to the provider to create any index, etc.
        vector_store = VectorStore(
            identifier=vector_store_id,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            provider_id=provider_id,
            provider_resource_id=vector_store_id,
            vector_store_name=params.name,
        )
        await self.register_vector_store(vector_store)

        # Create OpenAI vector store metadata
        status = "completed"

        # Start with no files attached and update later
        file_counts = VectorStoreFileCounts(
            cancelled=0,
            completed=0,
            failed=0,
            in_progress=0,
            total=0,
        )
        if not params.chunking_strategy or params.chunking_strategy.type == "auto":
            chunking_strategy = VectorStoreChunkingStrategyStatic(static=VectorStoreChunkingStrategyStaticConfig())
        else:
            chunking_strategy = params.chunking_strategy
        store_info: dict[str, Any] = {
            "id": vector_store_id,
            "object": "vector_store",
            "created_at": created_at,
            "name": params.name,
            "usage_bytes": 0,
            "file_counts": file_counts.model_dump(),
            "status": status,
            "expires_after": params.expires_after,
            "expires_at": None,
            "last_active_at": created_at,
            "file_ids": [],
            "chunking_strategy": chunking_strategy.model_dump(),
        }

        # Add provider information to metadata if provided
        if provider_id:
            metadata["provider_id"] = provider_id
        if provider_vector_store_id:
            metadata["provider_vector_store_id"] = provider_vector_store_id
        store_info["metadata"] = metadata

        # Save to persistent storage (provider-specific)
        await self._save_openai_vector_store(vector_store_id, store_info)

        # Store in memory cache
        self.openai_vector_stores[vector_store_id] = store_info

        # Now that our vector store is created, attach any files that were provided
        file_ids = params.file_ids or []
        tasks = [self.openai_attach_file_to_vector_store(vector_store_id, file_id) for file_id in file_ids]
        await asyncio.gather(*tasks)

        # Get the updated store info and return it
        store_info = self.openai_vector_stores[vector_store_id]
        return VectorStoreObject.model_validate(store_info)

    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        """Returns a list of vector stores."""
        limit = limit or 20
        order = order or "desc"

        # Get all vector stores
        all_stores = list(self.openai_vector_stores.values())

        # Sort by created_at
        reverse_order = order == "desc"
        all_stores.sort(key=lambda x: x["created_at"], reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, store in enumerate(all_stores) if store["id"] == after), -1)
            if after_index >= 0:
                all_stores = all_stores[after_index + 1 :]

        if before:
            before_index = next(
                (i for i, store in enumerate(all_stores) if store["id"] == before),
                len(all_stores),
            )
            all_stores = all_stores[:before_index]

        # Apply limit
        limited_stores = all_stores[:limit]
        # Convert to VectorStoreObject instances
        data = [VectorStoreObject(**store) for store in limited_stores]

        # Determine pagination info
        has_more = len(all_stores) > limit
        first_id = data[0].id if data else None
        last_id = data[-1].id if data else None

        return VectorStoreListResponse(
            data=data,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]
        return VectorStoreObject(**store_info)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        """Modifies a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id].copy()

        # Update fields if provided
        if name is not None:
            store_info["name"] = name
        if expires_after is not None:
            store_info["expires_after"] = expires_after
        if metadata is not None:
            store_info["metadata"] = metadata

        # Update last_active_at
        store_info["last_active_at"] = int(time.time())

        # Save to persistent storage (provider-specific)
        await self._update_openai_vector_store(vector_store_id, store_info)

        # Update in-memory cache
        self.openai_vector_stores[vector_store_id] = store_info

        return VectorStoreObject(**store_info)

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        # Delete from persistent storage (provider-specific)
        await self._delete_openai_vector_store_from_storage(vector_store_id)

        # Delete from in-memory cache
        self.openai_vector_stores.pop(vector_store_id, None)

        # Also delete the underlying vector DB
        try:
            await self.unregister_vector_store(vector_store_id)
        except Exception as e:
            logger.warning(f"Failed to delete underlying vector DB {vector_store_id}: {e}")

        return VectorStoreDeleteResponse(
            id=vector_store_id,
            deleted=True,
        )

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = 10,
        ranking_options: SearchRankingOptions | None = None,
        rewrite_query: bool | None = False,
        search_mode: (
            str | None
        ) = "vector",  # Using str instead of Literal due to OpenAPI schema generator limitations
    ) -> VectorStoreSearchResponsePage:
        """Search for chunks in a vector store."""
        max_num_results = max_num_results or 10

        # Validate search_mode
        valid_modes = {"keyword", "vector", "hybrid"}
        if search_mode not in valid_modes:
            raise ValueError(f"search_mode must be one of {valid_modes}, got {search_mode}")

        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        if isinstance(query, list):
            search_query = " ".join(query)
        else:
            search_query = query

        try:
            score_threshold = (
                ranking_options.score_threshold
                if ranking_options and ranking_options.score_threshold is not None
                else 0.0
            )
            params = {
                "max_chunks": max_num_results * CHUNK_MULTIPLIER,
                "score_threshold": score_threshold,
                "mode": search_mode,
            }
            # TODO: Add support for ranking_options.ranker

            response = await self.query_chunks(
                vector_store_id=vector_store_id,
                query=search_query,
                params=params,
            )

            # Convert response to OpenAI format
            data = []
            for chunk, score in zip(response.chunks, response.scores, strict=False):
                # Apply filters if provided
                if filters:
                    # Simple metadata filtering
                    if not self._matches_filters(chunk.metadata, filters):
                        continue

                content = self._chunk_to_vector_store_content(chunk)

                response_data_item = VectorStoreSearchResponse(
                    file_id=chunk.metadata.get("document_id", ""),
                    filename=chunk.metadata.get("filename", ""),
                    score=score,
                    attributes=chunk.metadata,
                    content=content,
                )
                data.append(response_data_item)
                if len(data) >= max_num_results:
                    break

            return VectorStoreSearchResponsePage(
                search_query=query if isinstance(query, list) else [query],
                data=data,
                has_more=False,  # For simplicity, we don't implement pagination here
                next_page=None,
            )

        except Exception as e:
            logger.error(f"Error searching vector store {vector_store_id}: {e}")
            # Return empty results on error
            return VectorStoreSearchResponsePage(
                search_query=query if isinstance(query, list) else [query],
                data=[],
                has_more=False,
                next_page=None,
            )

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        if not filters:
            return True

        filter_type = filters.get("type")

        if filter_type in ["eq", "ne", "gt", "gte", "lt", "lte"]:
            # Comparison filter
            key = filters.get("key")
            value = filters.get("value")

            if key not in metadata:
                return False

            metadata_value = metadata[key]

            if filter_type == "eq":
                return bool(metadata_value == value)
            elif filter_type == "ne":
                return bool(metadata_value != value)
            elif filter_type == "gt":
                return bool(metadata_value > value)
            elif filter_type == "gte":
                return bool(metadata_value >= value)
            elif filter_type == "lt":
                return bool(metadata_value < value)
            elif filter_type == "lte":
                return bool(metadata_value <= value)
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")

        elif filter_type == "and":
            # All filters must match
            sub_filters = filters.get("filters", [])
            return all(self._matches_filters(metadata, f) for f in sub_filters)

        elif filter_type == "or":
            # At least one filter must match
            sub_filters = filters.get("filters", [])
            return any(self._matches_filters(metadata, f) for f in sub_filters)

        else:
            # Unknown filter type, default to no match
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def _chunk_to_vector_store_content(
        self, chunk: Chunk, include_embeddings: bool = False, include_metadata: bool = False
    ) -> list[VectorStoreContent]:
        def extract_fields() -> dict:
            """Extract embedding and metadata fields from chunk based on include flags."""
            return {
                "embedding": chunk.embedding if include_embeddings else None,
                "chunk_metadata": chunk.chunk_metadata if include_metadata else None,
                "metadata": chunk.metadata if include_metadata else None,
            }

        fields = extract_fields()

        if isinstance(chunk.content, str):
            content_item = VectorStoreContent(type="text", text=chunk.content, **fields)
            content = [content_item]
        elif isinstance(chunk.content, list):
            # TODO: Add support for other types of content
            content = []
            for item in chunk.content:
                if item.type == "text":
                    content_item = VectorStoreContent(type="text", text=item.text, **fields)
                    content.append(content_item)
        else:
            if chunk.content.type != "text":
                raise ValueError(f"Unsupported content type: {chunk.content.type}")

            content_item = VectorStoreContent(type="text", text=chunk.content.text, **fields)
            content = [content_item]
        return content

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> VectorStoreFileObject:
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        # Check if file is already attached to this vector store
        store_info = self.openai_vector_stores[vector_store_id]
        if file_id in store_info["file_ids"]:
            logger.warning(f"File {file_id} is already attached to vector store {vector_store_id}, skipping")
            # Return existing file object
            file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
            return VectorStoreFileObject(**file_info)

        attributes = attributes or {}
        chunking_strategy = chunking_strategy or VectorStoreChunkingStrategyAuto()
        created_at = int(time.time())
        chunks: list[Chunk] = []
        file_response: OpenAIFileObject | None = None

        vector_store_file_object = VectorStoreFileObject(
            id=file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
            created_at=created_at,
            status="in_progress",
            vector_store_id=vector_store_id,
        )

        if not hasattr(self, "files_api") or not self.files_api:
            vector_store_file_object.status = "failed"
            vector_store_file_object.last_error = VectorStoreFileLastError(
                code="server_error",
                message="Files API is not available",
            )
            return vector_store_file_object

        if isinstance(chunking_strategy, VectorStoreChunkingStrategyStatic):
            max_chunk_size_tokens = chunking_strategy.static.max_chunk_size_tokens
            chunk_overlap_tokens = chunking_strategy.static.chunk_overlap_tokens
        else:
            # Default values from OpenAI API spec
            max_chunk_size_tokens = 800
            chunk_overlap_tokens = 400

        try:
            file_response = await self.files_api.openai_retrieve_file(file_id)
            mime_type, _ = mimetypes.guess_type(file_response.filename)
            content_response = await self.files_api.openai_retrieve_file_content(file_id)

            content = content_from_data_and_mime_type(content_response.body, mime_type)

            chunk_attributes = attributes.copy()
            chunk_attributes["filename"] = file_response.filename

            chunks = make_overlapped_chunks(
                file_id,
                content,
                max_chunk_size_tokens,
                chunk_overlap_tokens,
                chunk_attributes,
            )
            if not chunks:
                vector_store_file_object.status = "failed"
                vector_store_file_object.last_error = VectorStoreFileLastError(
                    code="server_error",
                    message="No chunks were generated from the file",
                )
            else:
                await self.insert_chunks(
                    vector_store_id=vector_store_id,
                    chunks=chunks,
                )
                vector_store_file_object.status = "completed"
        except Exception as e:
            logger.exception("Error attaching file to vector store")
            vector_store_file_object.status = "failed"
            vector_store_file_object.last_error = VectorStoreFileLastError(
                code="server_error",
                message=str(e),
            )

        # Save vector store file to persistent storage AFTER insert_chunks
        # so that chunks include the embeddings that were generated
        file_info = vector_store_file_object.model_dump(exclude={"last_error"})
        file_info["filename"] = file_response.filename if file_response else ""

        dict_chunks = [c.model_dump() for c in chunks]
        await self._save_openai_vector_store_file(vector_store_id, file_id, file_info, dict_chunks)

        # Update file_ids and file_counts in vector store metadata
        store_info = self.openai_vector_stores[vector_store_id].copy()
        store_info["file_ids"].append(file_id)
        store_info["file_counts"]["total"] += 1
        store_info["file_counts"][vector_store_file_object.status] += 1

        # Save updated vector store to persistent storage
        await self._save_openai_vector_store(vector_store_id, store_info)

        # Update vector store in-memory cache
        self.openai_vector_stores[vector_store_id] = store_info

        return vector_store_file_object

    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        """List files in a vector store."""
        limit = limit or 20
        order = order or "desc"

        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]

        file_objects: list[VectorStoreFileObject] = []
        for file_id in store_info["file_ids"]:
            file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
            file_object = VectorStoreFileObject(**file_info)
            if filter and file_object.status != filter:
                continue
            file_objects.append(file_object)

        # Sort by created_at
        reverse_order = order == "desc"
        file_objects.sort(key=lambda x: x.created_at, reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, file in enumerate(file_objects) if file.id == after), -1)
            if after_index >= 0:
                file_objects = file_objects[after_index + 1 :]

        if before:
            before_index = next(
                (i for i, file in enumerate(file_objects) if file.id == before),
                len(file_objects),
            )
            file_objects = file_objects[:before_index]

        # Apply limit
        limited_files = file_objects[:limit]

        # Determine pagination info
        has_more = len(file_objects) > limit
        first_id = limited_files[0].id if file_objects else None
        last_id = limited_files[-1].id if file_objects else None

        return VectorStoreListFilesResponse(
            data=limited_files,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        """Retrieves a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]
        if file_id not in store_info["file_ids"]:
            raise ValueError(f"File {file_id} not found in vector store {vector_store_id}")

        file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
        return VectorStoreFileObject(**file_info)

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
        include_embeddings: bool | None = False,
        include_metadata: bool | None = False,
    ) -> VectorStoreFileContentResponse:
        """Retrieves the contents of a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        # Parameters are already provided directly
        # include_embeddings and include_metadata are now function parameters

        dict_chunks = await self._load_openai_vector_store_file_contents(vector_store_id, file_id)
        chunks = [Chunk.model_validate(c) for c in dict_chunks]
        content = []
        for chunk in chunks:
            content.extend(
                self._chunk_to_vector_store_content(
                    chunk, include_embeddings=include_embeddings or False, include_metadata=include_metadata or False
                )
            )
        return VectorStoreFileContentResponse(
            data=content,
        )

    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any],
    ) -> VectorStoreFileObject:
        """Updates a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]
        if file_id not in store_info["file_ids"]:
            raise ValueError(f"File {file_id} not found in vector store {vector_store_id}")

        file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
        file_info["attributes"] = attributes
        await self._update_openai_vector_store_file(vector_store_id, file_id, file_info)
        return VectorStoreFileObject(**file_info)

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        """Deletes a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        dict_chunks = await self._load_openai_vector_store_file_contents(vector_store_id, file_id)
        chunks = [Chunk.model_validate(c) for c in dict_chunks]

        # Create ChunkForDeletion objects with both chunk_id and document_id
        chunks_for_deletion = []
        for c in chunks:
            if c.chunk_id:
                document_id = c.metadata.get("document_id") or (
                    c.chunk_metadata.document_id if c.chunk_metadata else None
                )
                if document_id:
                    chunks_for_deletion.append(ChunkForDeletion(chunk_id=str(c.chunk_id), document_id=document_id))
                else:
                    logger.warning(f"Chunk {c.chunk_id} has no document_id, skipping deletion")

        if chunks_for_deletion:
            await self.delete_chunks(vector_store_id, chunks_for_deletion)

        store_info = self.openai_vector_stores[vector_store_id].copy()

        file = await self.openai_retrieve_vector_store_file(vector_store_id, file_id)
        await self._delete_openai_vector_store_file_from_storage(vector_store_id, file_id)

        # Update in-memory cache
        store_info["file_ids"].remove(file_id)
        store_info["file_counts"][file.status] -= 1
        store_info["file_counts"]["total"] -= 1
        self.openai_vector_stores[vector_store_id] = store_info

        # Save updated vector store to persistent storage
        await self._save_openai_vector_store(vector_store_id, store_info)

        return VectorStoreFileDeleteResponse(
            id=file_id,
            deleted=True,
        )

    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        """Create a vector store file batch."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        chunking_strategy = params.chunking_strategy or VectorStoreChunkingStrategyAuto()

        created_at = int(time.time())
        batch_id = generate_object_id("vector_store_file_batch", lambda: f"batch_{uuid.uuid4()}")
        # File batches expire after 7 days
        expires_at = created_at + (7 * 24 * 60 * 60)

        # Initialize batch file counts - all files start as in_progress
        file_counts = VectorStoreFileCounts(
            completed=0,
            cancelled=0,
            failed=0,
            in_progress=len(params.file_ids),
            total=len(params.file_ids),
        )

        # Create batch object immediately with in_progress status
        batch_object = VectorStoreFileBatchObject(
            id=batch_id,
            created_at=created_at,
            vector_store_id=vector_store_id,
            status="in_progress",
            file_counts=file_counts,
        )

        batch_info = {
            **batch_object.model_dump(),
            "file_ids": params.file_ids,
            "attributes": params.attributes,
            "chunking_strategy": chunking_strategy.model_dump(),
            "expires_at": expires_at,
        }
        await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        # Start background processing of files
        task = asyncio.create_task(self._process_file_batch_async(batch_id, batch_info))
        self._file_batch_tasks[batch_id] = task

        # Run cleanup if needed (throttled to once every 1 day)
        current_time = int(time.time())
        if current_time - self._last_file_batch_cleanup_time >= FILE_BATCH_CLEANUP_INTERVAL_SECONDS:
            logger.info("Running throttled cleanup of expired file batches")
            asyncio.create_task(self._cleanup_expired_file_batches())
            self._last_file_batch_cleanup_time = current_time

        return batch_object

    async def _process_files_with_concurrency(
        self,
        file_ids: list[str],
        vector_store_id: str,
        attributes: dict[str, Any],
        chunking_strategy_obj: Any,
        batch_id: str,
        batch_info: dict[str, Any],
    ) -> None:
        """Process files with controlled concurrency and chunking."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES_PER_BATCH)

        async def process_single_file(file_id: str) -> tuple[str, bool]:
            """Process a single file with concurrency control."""
            async with semaphore:
                try:
                    vector_store_file_object = await self.openai_attach_file_to_vector_store(
                        vector_store_id=vector_store_id,
                        file_id=file_id,
                        attributes=attributes,
                        chunking_strategy=chunking_strategy_obj,
                    )
                    return file_id, vector_store_file_object.status == "completed"
                except Exception as e:
                    logger.error(f"Failed to process file {file_id} in batch {batch_id}: {e}")
                    return file_id, False

        # Process files in chunks to avoid creating too many tasks at once
        total_files = len(file_ids)
        for chunk_start in range(0, total_files, FILE_BATCH_CHUNK_SIZE):
            chunk_end = min(chunk_start + FILE_BATCH_CHUNK_SIZE, total_files)
            chunk = file_ids[chunk_start:chunk_end]

            chunk_num = chunk_start // FILE_BATCH_CHUNK_SIZE + 1
            total_chunks = (total_files + FILE_BATCH_CHUNK_SIZE - 1) // FILE_BATCH_CHUNK_SIZE
            logger.info(
                f"Processing chunk {chunk_num} of {total_chunks} ({len(chunk)} files, {chunk_start + 1}-{chunk_end} of {total_files} total files)"
            )

            async with asyncio.TaskGroup() as tg:
                chunk_tasks = [tg.create_task(process_single_file(file_id)) for file_id in chunk]

            chunk_results = [task.result() for task in chunk_tasks]

            # Update counts after each chunk for progressive feedback
            for _, success in chunk_results:
                self._update_file_counts(batch_info, success=success)

            # Save progress after each chunk
            await self._save_openai_vector_store_file_batch(batch_id, batch_info)

    def _update_file_counts(self, batch_info: dict[str, Any], success: bool) -> None:
        """Update file counts based on processing result."""
        if success:
            batch_info["file_counts"]["completed"] += 1
        else:
            batch_info["file_counts"]["failed"] += 1
        batch_info["file_counts"]["in_progress"] -= 1

    def _update_batch_status(self, batch_info: dict[str, Any]) -> None:
        """Update final batch status based on file processing results."""
        if batch_info["file_counts"]["failed"] == 0:
            batch_info["status"] = "completed"
        elif batch_info["file_counts"]["completed"] == 0:
            batch_info["status"] = "failed"
        else:
            batch_info["status"] = "completed"  # Partial success counts as completed

    async def _process_file_batch_async(
        self,
        batch_id: str,
        batch_info: dict[str, Any],
        override_file_ids: list[str] | None = None,
    ) -> None:
        """Process files in a batch asynchronously in the background."""
        file_ids = override_file_ids if override_file_ids is not None else batch_info["file_ids"]
        attributes = batch_info["attributes"]
        chunking_strategy = batch_info["chunking_strategy"]
        vector_store_id = batch_info["vector_store_id"]
        chunking_strategy_adapter: TypeAdapter[VectorStoreChunkingStrategy] = TypeAdapter(VectorStoreChunkingStrategy)
        chunking_strategy_obj = chunking_strategy_adapter.validate_python(chunking_strategy)

        try:
            # Process all files with controlled concurrency
            await self._process_files_with_concurrency(
                file_ids=file_ids,
                vector_store_id=vector_store_id,
                attributes=attributes,
                chunking_strategy_obj=chunking_strategy_obj,
                batch_id=batch_id,
                batch_info=batch_info,
            )

            # Update final batch status
            self._update_batch_status(batch_info)
            await self._save_openai_vector_store_file_batch(batch_id, batch_info)

            logger.info(f"File batch {batch_id} processing completed with status: {batch_info['status']}")

        except asyncio.CancelledError:
            logger.info(f"File batch {batch_id} processing was cancelled")
            # Clean up task reference if it still exists
            self._file_batch_tasks.pop(batch_id, None)
            raise  # Re-raise to ensure proper cancellation propagation
        finally:
            # Always clean up task reference when processing ends
            self._file_batch_tasks.pop(batch_id, None)

    def _get_and_validate_batch(self, batch_id: str, vector_store_id: str) -> dict[str, Any]:
        """Get and validate batch exists and belongs to vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        if batch_id not in self.openai_file_batches:
            raise ValueError(f"File batch {batch_id} not found")

        batch_info = self.openai_file_batches[batch_id]

        # Check if batch has expired (read-only check)
        expires_at = batch_info.get("expires_at")
        if expires_at:
            current_time = int(time.time())
            if current_time > expires_at:
                raise ValueError(f"File batch {batch_id} has expired after 7 days from creation")

        if batch_info["vector_store_id"] != vector_store_id:
            raise ValueError(f"File batch {batch_id} does not belong to vector store {vector_store_id}")

        return batch_info

    def _paginate_objects(
        self,
        objects: list[Any],
        limit: int | None = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> tuple[list[Any], bool, str | None, str | None]:
        """Apply pagination to a list of objects with id fields."""
        limit = min(limit or 20, 100)  # Cap at 100 as per OpenAI

        # Find start index
        start_idx = 0
        if after:
            for i, obj in enumerate(objects):
                if obj.id == after:
                    start_idx = i + 1
                    break

        # Find end index
        end_idx = start_idx + limit
        if before:
            for i, obj in enumerate(objects[start_idx:], start_idx):
                if obj.id == before:
                    end_idx = i
                    break

        # Apply pagination
        paginated_objects = objects[start_idx:end_idx]

        # Determine pagination info
        has_more = end_idx < len(objects)
        first_id = paginated_objects[0].id if paginated_objects else None
        last_id = paginated_objects[-1].id if paginated_objects else None

        return paginated_objects, has_more, first_id, last_id

    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Retrieve a vector store file batch."""
        batch_info = self._get_and_validate_batch(batch_id, vector_store_id)
        return VectorStoreFileBatchObject(**batch_info)

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
        """Returns a list of vector store files in a batch."""
        batch_info = self._get_and_validate_batch(batch_id, vector_store_id)
        batch_file_ids = batch_info["file_ids"]

        # Load file objects for files in this batch
        batch_file_objects = []

        for file_id in batch_file_ids:
            try:
                file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
                file_object = VectorStoreFileObject(**file_info)

                # Apply status filter if provided
                if filter and file_object.status != filter:
                    continue

                batch_file_objects.append(file_object)
            except Exception as e:
                logger.warning(f"Could not load file {file_id} from batch {batch_id}: {e}")
                continue

        # Sort by created_at
        reverse_order = order == "desc"
        batch_file_objects.sort(key=lambda x: x.created_at, reverse=reverse_order)

        # Apply pagination using helper
        paginated_files, has_more, first_id, last_id = self._paginate_objects(batch_file_objects, limit, after, before)

        return VectorStoreFilesListInBatchResponse(
            data=paginated_files,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more,
        )

    async def openai_cancel_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Cancel a vector store file batch."""
        batch_info = self._get_and_validate_batch(batch_id, vector_store_id)

        if batch_info["status"] not in ["in_progress"]:
            raise ValueError(f"Cannot cancel batch {batch_id} with status {batch_info['status']}")

        # Cancel the actual processing task if it exists
        if batch_id in self._file_batch_tasks:
            task = self._file_batch_tasks[batch_id]
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled processing task for file batch: {batch_id}")
            # Remove from task tracking
            del self._file_batch_tasks[batch_id]

        batch_info["status"] = "cancelled"

        await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        updated_batch = VectorStoreFileBatchObject(**batch_info)

        return updated_batch
