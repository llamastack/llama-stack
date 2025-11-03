# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.apis.common.content_types import URL
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import ListToolDefsResponse, ListToolGroupsResponse, ToolDef, ToolGroup


class ToolStore(Protocol):
    async def get_tool(self, tool_name: str) -> ToolDef: ...

    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup: ...


@runtime_checkable
@trace_protocol
class ToolGroupsService(Protocol):
    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool group."""
        ...

    async def get_tool_group(
        self,
        toolgroup_id: str,
    ) -> ToolGroup:
        """Get a tool group by its ID."""
        ...

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        """List tool groups with optional provider."""
        ...

    async def list_tools(self, toolgroup_id: str | None = None) -> ListToolDefsResponse:
        """List tools with optional tool group."""
        ...

    async def get_tool(
        self,
        tool_name: str,
    ) -> ToolDef:
        """Get a tool by its name."""
        ...

    async def unregister_toolgroup(
        self,
        toolgroup_id: str,
    ) -> None:
        """Unregister a tool group."""
        ...
