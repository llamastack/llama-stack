# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.log import get_logger
from llama_stack_api import (
    URL,
    ListToolDefsResponse,
    ToolRuntime,
)

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

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None) -> Any:
        logger.debug(f"ToolRuntimeRouter.invoke_tool: {tool_name}")
        provider = await self.routing_table.get_provider_impl(tool_name)
        return await provider.invoke_tool(
            tool_name=tool_name,
            kwargs=kwargs,
            authorization=authorization,
        )

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None, authorization: str | None = None
    ) -> ListToolDefsResponse:
        return await self.routing_table.list_tools(tool_group_id, authorization=authorization)
