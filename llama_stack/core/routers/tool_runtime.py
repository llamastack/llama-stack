# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import (
    URL,
    InterleavedContent,
)
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    ToolRuntime,
)
from llama_stack.log import get_logger

from ..routing_tables.toolgroups import ToolGroupsRoutingTable

logger = get_logger(name=__name__, category="core::routers")


class ToolRuntimeRouter(ToolRuntime):
    class RagToolImpl(RAGToolRuntime):
        def __init__(
            self,
            routing_table: ToolGroupsRoutingTable,
        ) -> None:
            logger.debug("Initializing ToolRuntimeRouter.RagToolImpl")
            self.routing_table = routing_table

        async def query(
            self,
            content: InterleavedContent,
            vector_db_ids: list[str],
            query_config: RAGQueryConfig | None = None,
        ) -> RAGQueryResult:
            logger.debug(f"ToolRuntimeRouter.RagToolImpl.query: {vector_db_ids}")
            provider = await self.routing_table.get_provider_impl("knowledge_search")
            return await provider.query(content, vector_db_ids, query_config)

        async def insert(
            self,
            documents: list[RAGDocument],
            vector_db_id: str,
            chunk_size_in_tokens: int = 512,
        ) -> None:
            logger.debug(
                f"ToolRuntimeRouter.RagToolImpl.insert: {vector_db_id}, {len(documents)} documents, chunk_size={chunk_size_in_tokens}"
            )
            provider = await self.routing_table.get_provider_impl("insert_into_memory")
            return await provider.insert(documents, vector_db_id, chunk_size_in_tokens)

    def __init__(
        self,
        routing_table: ToolGroupsRoutingTable,
    ) -> None:
        logger.debug("Initializing ToolRuntimeRouter")
        self.routing_table = routing_table

        # HACK ALERT this should be in sync with "get_all_api_endpoints()"
        self.rag_tool = self.RagToolImpl(routing_table)
        for method in ("query", "insert"):
            setattr(self, f"rag_tool.{method}", getattr(self.rag_tool, method))

    async def initialize(self) -> None:
        logger.debug("ToolRuntimeRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("ToolRuntimeRouter.shutdown")
        pass

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        logger.debug(f"ToolRuntimeRouter.invoke_tool: {tool_name}")
        provider = await self.routing_table.get_provider_impl(tool_name)
        return await provider.invoke_tool(
            tool_name=tool_name,
            kwargs=kwargs,
        )

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        logger.debug(f"ToolRuntimeRouter.list_runtime_tools: {tool_group_id}")
        return await self.routing_table.list_tools(tool_group_id)
