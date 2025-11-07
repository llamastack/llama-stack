# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import (
    URL,
)
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolRuntime,
)
from llama_stack.log import get_logger

from ..routing_tables.toolgroups import ToolGroupsRoutingTable

logger = get_logger(name=__name__, category="core::routers")


class ToolRuntimeRouter(ToolRuntime):
    def __init__(
        self,
        routing_table: ToolGroupsRoutingTable,
    ) -> None:
        logger.debug("Initializing ToolRuntimeRouter")
        self.routing_table = routing_table

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
