# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import mimetypes
from typing import Any

import httpx
from fastapi import UploadFile
from pydantic import TypeAdapter

from llama_stack.apis.common.content_types import (
    URL,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.files import Files, OpenAIFilePurpose
from llama_stack.apis.inference import Inference
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    ToolDef,
    ToolGroup,
    ToolInvocationResult,
    ToolRuntime,
)
from llama_stack.apis.vector_io import (
    QueryChunksResponse,
    VectorIO,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.memory.vector_store import parse_data_url

from .config import RagToolRuntimeConfig
from .context_retriever import generate_rag_query

log = get_logger(name=__name__, category="tool_runtime")


async def raw_data_from_doc(doc: RAGDocument) -> tuple[bytes, str]:
    """Get raw binary data and mime type from a RAGDocument for file upload."""
    if isinstance(doc.content, URL):
        if doc.content.uri.startswith("data:"):
            parts = parse_data_url(doc.content.uri)
            mime_type = parts["mimetype"]
            data = parts["data"]

            if parts["is_base64"]:
                file_data = base64.b64decode(data)
            else:
                file_data = data.encode("utf-8")

            return file_data, mime_type
        else:
            async with httpx.AsyncClient() as client:
                r = await client.get(doc.content.uri)
                r.raise_for_status()
                mime_type = r.headers.get("content-type", "application/octet-stream")
                return r.content, mime_type
    else:
        if isinstance(doc.content, str):
            content_str = doc.content
        else:
            content_str = interleaved_content_as_str(doc.content)

        if content_str.startswith("data:"):
            parts = parse_data_url(content_str)
            mime_type = parts["mimetype"]
            data = parts["data"]

            if parts["is_base64"]:
                file_data = base64.b64decode(data)
            else:
                file_data = data.encode("utf-8")

            return file_data, mime_type
        else:
            return content_str.encode("utf-8"), "text/plain"


class MemoryToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime):
    def __init__(
        self,
        config: RagToolRuntimeConfig,
        vector_io_api: VectorIO,
        inference_api: Inference,
        files_api: Files,
    ):
        self.config = config
        self.vector_io_api = vector_io_api
        self.inference_api = inference_api
        self.files_api = files_api

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def insert(
        self,
        documents: list[RAGDocument],
        vector_store_id: str,
        chunk_size_in_tokens: int = 512,
    ) -> None:
        if not documents:
            return

        for doc in documents:
            try:
                try:
                    file_data, mime_type = await raw_data_from_doc(doc)
                except Exception as e:
                    log.error(f"Failed to extract content from document {doc.document_id}: {e}")
                    continue

                file_extension = mimetypes.guess_extension(mime_type) or ".txt"
                filename = doc.metadata.get("filename", f"{doc.document_id}{file_extension}")

                file_obj = io.BytesIO(file_data)
                file_obj.name = filename

                upload_file = UploadFile(file=file_obj, filename=filename)

                try:
                    created_file = await self.files_api.openai_upload_file(
                        file=upload_file, purpose=OpenAIFilePurpose.ASSISTANTS
                    )
                except Exception as e:
                    log.error(f"Failed to upload file for document {doc.document_id}: {e}")
                    continue

                chunking_strategy = VectorStoreChunkingStrategyStatic(
                    static=VectorStoreChunkingStrategyStaticConfig(
                        max_chunk_size_tokens=chunk_size_in_tokens,
                        chunk_overlap_tokens=chunk_size_in_tokens // 4,
                    )
                )

                try:
                    await self.vector_io_api.openai_attach_file_to_vector_store(
                        vector_store_id=vector_store_id,
                        file_id=created_file.id,
                        attributes=doc.metadata,
                        chunking_strategy=chunking_strategy,
                    )
                except Exception as e:
                    log.error(
                        f"Failed to attach file {created_file.id} to vector store {vector_store_id} for document {doc.document_id}: {e}"
                    )
                    continue

            except Exception as e:
                log.error(f"Unexpected error processing document {doc.document_id}: {e}")
                continue

    async def query(
        self,
        content: InterleavedContent,
        vector_store_ids: list[str],
        query_config: RAGQueryConfig | None = None,
    ) -> RAGQueryResult:
        if not vector_store_ids:
            raise ValueError(
                "No vector DBs were provided to the knowledge search tool. Please provide at least one vector DB ID."
            )

        query_config = query_config or RAGQueryConfig()
        query = await generate_rag_query(
            query_config.query_generator_config,
            content,
            inference_api=self.inference_api,
        )
        tasks = [
            self.vector_io_api.query_chunks(
                vector_store_id=vector_store_id,
                query=query,
                params={
                    "mode": query_config.mode,
                    "max_chunks": query_config.max_chunks,
                    "score_threshold": 0.0,
                    "ranker": query_config.ranker,
                },
            )
            for vector_store_id in vector_store_ids
        ]
        results: list[QueryChunksResponse] = await asyncio.gather(*tasks)

        chunks = []
        scores = []

        for vector_store_id, result in zip(vector_store_ids, results, strict=False):
            for chunk, score in zip(result.chunks, result.scores, strict=False):
                if not hasattr(chunk, "metadata") or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["vector_store_id"] = vector_store_id

                chunks.append(chunk)
                scores.append(score)

        if not chunks:
            return RAGQueryResult(content=None)

        # sort by score
        chunks, scores = zip(*sorted(zip(chunks, scores, strict=False), key=lambda x: x[1], reverse=True), strict=False)  # type: ignore
        chunks = chunks[: query_config.max_chunks]

        tokens = 0
        picked: list[InterleavedContentItem] = [
            TextContentItem(
                text=f"knowledge_search tool found {len(chunks)} chunks:\nBEGIN of knowledge_search tool results.\n"
            )
        ]
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            tokens += metadata.get("token_count", 0)
            tokens += metadata.get("metadata_token_count", 0)

            if tokens > query_config.max_tokens_in_context:
                log.error(
                    f"Using {len(picked)} chunks; reached max tokens in context: {tokens}",
                )
                break

            # Add useful keys from chunk_metadata to metadata and remove some from metadata
            chunk_metadata_keys_to_include_from_context = [
                "chunk_id",
                "document_id",
                "source",
            ]
            metadata_keys_to_exclude_from_context = [
                "token_count",
                "metadata_token_count",
                "vector_store_id",
            ]
            metadata_for_context = {}
            for k in chunk_metadata_keys_to_include_from_context:
                metadata_for_context[k] = getattr(chunk.chunk_metadata, k)
            for k in metadata:
                if k not in metadata_keys_to_exclude_from_context:
                    metadata_for_context[k] = metadata[k]

            text_content = query_config.chunk_template.format(index=i + 1, chunk=chunk, metadata=metadata_for_context)
            picked.append(TextContentItem(text=text_content))

        picked.append(TextContentItem(text="END of knowledge_search tool results.\n"))
        picked.append(
            TextContentItem(
                text=f'The above results were retrieved to help answer the user\'s query: "{interleaved_content_as_str(content)}". Use them as supporting information only in answering this query.\n',
            )
        )

        return RAGQueryResult(
            content=picked,
            metadata={
                "document_ids": [c.document_id for c in chunks[: len(picked)]],
                "chunks": [c.content for c in chunks[: len(picked)]],
                "scores": scores[: len(picked)],
                "vector_store_ids": [c.metadata["vector_store_id"] for c in chunks[: len(picked)]],
            },
        )

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        # Parameters are not listed since these methods are not yet invoked automatically
        # by the LLM. The method is only implemented so things like /tools can list without
        # encountering fatals.
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="insert_into_memory",
                    description="Insert documents into memory",
                ),
                ToolDef(
                    name="knowledge_search",
                    description="Search for information in a database.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for. Can be a natural language sentence or keywords.",
                            }
                        },
                        "required": ["query"],
                    },
                ),
            ]
        )

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        vector_store_ids = kwargs.get("vector_store_ids", [])
        query_config = kwargs.get("query_config")
        if query_config:
            query_config = TypeAdapter(RAGQueryConfig).validate_python(query_config)
        else:
            query_config = RAGQueryConfig()

        query = kwargs["query"]
        result = await self.query(
            content=query,
            vector_store_ids=vector_store_ids,
            query_config=query_config,
        )

        return ToolInvocationResult(
            content=result.content or [],
            metadata={
                **(result.metadata or {}),
                "citation_files": getattr(result, "citation_files", None),
            },
        )
