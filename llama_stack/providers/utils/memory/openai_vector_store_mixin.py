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
from typing import Any

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files import Files, OpenAIFileObject
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreContent,
    VectorStoreDeleteResponse,
    VectorStoreFileContentsResponse,
    VectorStoreFileCounts,
    VectorStoreFileDeleteResponse,
    VectorStoreFileLastError,
    VectorStoreFileObject,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    content_from_data_and_mime_type,
    make_overlapped_chunks,
)

logger = get_logger(name=__name__, category="providers::utils")

# Constants for OpenAI vector stores
CHUNK_MULTIPLIER = 5

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_dbs:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:{VERSION}::"


class OpenAIVectorStoreMixin(ABC):
    """
    Mixin class that provides common OpenAI Vector Store API implementation.
    Providers need to implement the abstract storage methods and maintain
    an openai_vector_stores in-memory cache.
    """

    # These should be provided by the implementing class
    openai_vector_stores: dict[str, dict[str, Any]]
    files_api: Files | None
    # KV store for persisting OpenAI vector store metadata
    kvstore: KVStore | None

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
        self, store_id: str, file_id: str, file_info: dict[str, Any], file_contents: list[dict[str, Any]]
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

    async def initialize_openai_vector_stores(self) -> None:
        """Load existing OpenAI vector stores into the in-memory cache."""
        self.openai_vector_stores = await self._load_openai_vector_stores()

    @abstractmethod
    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from a vector store."""
        pass

    @abstractmethod
    async def register_vector_db(self, vector_db: VectorDB) -> None:
        """Register a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def unregister_vector_db(self, vector_db_id: str) -> None:
        """Unregister a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert chunks into a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def query_chunks(
        self, vector_db_id: str, query: Any, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        """Query chunks from a vector database (provider-specific implementation)."""
        pass

    async def openai_create_vector_store(
        self,
        name: str | None = None,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorStoreObject:
        """Creates a vector store."""
        created_at = int(time.time())
        # Derive the canonical vector_db_id (allow override, else generate)
        vector_db_id = provider_vector_db_id or f"vs_{uuid.uuid4()}"

        if provider_id is None:
            raise ValueError("Provider ID is required")

        if embedding_model is None:
            raise ValueError("Embedding model is required")

        # Embedding dimension is required (defaulted to 384 if not provided)
        if embedding_dimension is None:
            raise ValueError("Embedding dimension is required")

        # Register the VectorDB backing this vector store
        vector_db = VectorDB(
            identifier=vector_db_id,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            provider_id=provider_id,
            provider_resource_id=vector_db_id,
            vector_db_name=name,
        )
        await self.register_vector_db(vector_db)

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
        store_info: dict[str, Any] = {
            "id": vector_db_id,
            "object": "vector_store",
            "created_at": created_at,
            "name": name,
            "usage_bytes": 0,
            "file_counts": file_counts.model_dump(),
            "status": status,
            "expires_after": expires_after,
            "expires_at": None,
            "last_active_at": created_at,
            "file_ids": [],
            "chunking_strategy": chunking_strategy,
        }

        # Add provider information to metadata if provided
        metadata = metadata or {}
        if provider_id:
            metadata["provider_id"] = provider_id
        if provider_vector_db_id:
            metadata["provider_vector_db_id"] = provider_vector_db_id
        store_info["metadata"] = metadata

        # Save to persistent storage (provider-specific)
        await self._save_openai_vector_store(vector_db_id, store_info)

        # Store in memory cache
        self.openai_vector_stores[vector_db_id] = store_info

        # Now that our vector store is created, attach any files that were provided
        file_ids = file_ids or []
        tasks = [self.openai_attach_file_to_vector_store(vector_db_id, file_id) for file_id in file_ids]
        await asyncio.gather(*tasks)

        # Get the updated store info and return it
        store_info = self.openai_vector_stores[vector_db_id]
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
            before_index = next((i for i, store in enumerate(all_stores) if store["id"] == before), len(all_stores))
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
            await self.unregister_vector_db(vector_store_id)
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
        search_mode: str | None = "vector",  # Using str instead of Literal due to OpenAPI schema generator limitations
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
                vector_db_id=vector_store_id,
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
                    file_id=chunk.metadata.get("file_id", ""),
                    filename=chunk.metadata.get("filename", ""),
                    score=score,
                    attributes=chunk.metadata,
                    content=content,
                )
                data.append(response_data_item)
                if len(data) >= max_num_results:
                    break

            return VectorStoreSearchResponsePage(
                search_query=search_query,
                data=data,
                has_more=False,  # For simplicity, we don't implement pagination here
                next_page=None,
            )

        except Exception as e:
            logger.error(f"Error searching vector store {vector_store_id}: {e}")
            # Return empty results on error
            return VectorStoreSearchResponsePage(
                search_query=search_query,
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

    def _chunk_to_vector_store_content(self, chunk: Chunk) -> list[VectorStoreContent]:
        # content is InterleavedContent
        if isinstance(chunk.content, str):
            content = [
                VectorStoreContent(
                    type="text",
                    text=chunk.content,
                )
            ]
        elif isinstance(chunk.content, list):
            # TODO: Add support for other types of content
            content = [
                VectorStoreContent(
                    type="text",
                    text=item.text,
                )
                for item in chunk.content
                if item.type == "text"
            ]
        else:
            if chunk.content.type != "text":
                raise ValueError(f"Unsupported content type: {chunk.content.type}")
            content = [
                VectorStoreContent(
                    type="text",
                    text=chunk.content.text,
                )
            ]
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

            chunks = make_overlapped_chunks(
                file_id,
                content,
                max_chunk_size_tokens,
                chunk_overlap_tokens,
                attributes,
            )

            if not chunks:
                vector_store_file_object.status = "failed"
                vector_store_file_object.last_error = VectorStoreFileLastError(
                    code="server_error",
                    message="No chunks were generated from the file",
                )
            else:
                await self.insert_chunks(
                    vector_db_id=vector_store_id,
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

        # Create OpenAI vector store file metadata
        file_info = vector_store_file_object.model_dump(exclude={"last_error"})
        file_info["filename"] = file_response.filename if file_response else ""

        # Save vector store file to persistent storage (provider-specific)
        dict_chunks = [c.model_dump() for c in chunks]
        # This should be updated to include chunk_id
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
            before_index = next((i for i, file in enumerate(file_objects) if file.id == before), len(file_objects))
            file_objects = file_objects[:before_index]

        # Apply limit
        limited_files = file_objects[:limit]

        # Determine pagination info
        has_more = len(file_objects) > limit
        first_id = file_objects[0].id if file_objects else None
        last_id = file_objects[-1].id if file_objects else None

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
    ) -> VectorStoreFileContentsResponse:
        """Retrieves the contents of a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
        dict_chunks = await self._load_openai_vector_store_file_contents(vector_store_id, file_id)
        chunks = [Chunk.model_validate(c) for c in dict_chunks]
        content = []
        for chunk in chunks:
            content.extend(self._chunk_to_vector_store_content(chunk))
        return VectorStoreFileContentsResponse(
            file_id=file_id,
            filename=file_info.get("filename", ""),
            attributes=file_info.get("attributes", {}),
            content=content,
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
