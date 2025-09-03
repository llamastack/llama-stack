# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import secrets
import string
from typing import Any

from pydantic import TypeAdapter

from llama_stack.apis.common.content_types import (
    URL,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import Inference
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    ToolDef,
    ToolGroup,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.apis.vector_io import QueryChunksResponse, VectorIO
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.memory.vector_store import (
    content_from_doc,
    make_overlapped_chunks,
)

from .config import RagToolRuntimeConfig
from .context_retriever import generate_rag_query

log = get_logger(name=__name__, category="tool_runtime")


def make_random_string(length: int = 8):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


class MemoryToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, RAGToolRuntime):
    def __init__(
        self,
        config: RagToolRuntimeConfig,
        vector_io_api: VectorIO,
        inference_api: Inference,
    ):
        self.config = config
        self.vector_io_api = vector_io_api
        self.inference_api = inference_api

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
        vector_db_id: str,
        chunk_size_in_tokens: int = 512,
    ) -> None:
        chunks = []
        for doc in documents:
            content = await content_from_doc(doc)
            # TODO: we should add enrichment here as URLs won't be added to the metadata by default
            chunks.extend(
                make_overlapped_chunks(
                    doc.document_id,
                    content,
                    chunk_size_in_tokens,
                    chunk_size_in_tokens // 4,
                    doc.metadata,
                )
            )

        if not chunks:
            return

        await self.vector_io_api.insert_chunks(
            chunks=chunks,
            vector_db_id=vector_db_id,
        )

    async def query(
        self,
        content: InterleavedContent,
        vector_db_ids: list[str],
        query_config: RAGQueryConfig | None = None,
    ) -> RAGQueryResult:
        if not vector_db_ids:
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
                vector_db_id=vector_db_id,
                query=query,
                params={
                    "mode": query_config.mode,
                    "max_chunks": query_config.max_chunks,
                    "score_threshold": 0.0,
                    "ranker": query_config.ranker,
                },
            )
            for vector_db_id in vector_db_ids
        ]
        results: list[QueryChunksResponse] = await asyncio.gather(*tasks)

        chunks = []
        scores = []

        for vector_db_id, result in zip(vector_db_ids, results, strict=False):
            for chunk, score in zip(result.chunks, result.scores, strict=False):
                if not hasattr(chunk, "metadata") or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["vector_db_id"] = vector_db_id

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
                "vector_db_id",
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
                "document_ids": [c.metadata["document_id"] for c in chunks[: len(picked)]],
                "chunks": [c.content for c in chunks[: len(picked)]],
                "scores": scores[: len(picked)],
                "vector_db_ids": [c.metadata["vector_db_id"] for c in chunks[: len(picked)]],
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
                    parameters=[
                        ToolParameter(
                            name="query",
                            description="The query to search for. Can be a natural language sentence or keywords.",
                            parameter_type="string",
                        ),
                    ],
                ),
            ]
        )

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        vector_db_ids = kwargs.get("vector_db_ids", [])
        query_config = kwargs.get("query_config")
        if query_config:
            query_config = TypeAdapter(RAGQueryConfig).validate_python(query_config)
        else:
            # handle someone passing an empty dict
            query_config = RAGQueryConfig()

        query = kwargs["query"]
        result = await self.query(
            content=query,
            vector_db_ids=vector_db_ids,
            query_config=query_config,
        )

        return ToolInvocationResult(
            content=result.content,
            metadata=result.metadata,
        )
