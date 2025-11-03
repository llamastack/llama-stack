# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.apis.common.content_types import URL
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import ListToolDefsResponse, ToolInvocationResult
from .rag_tool import RAGToolRuntime


class ToolStore(Protocol):
    async def get_tool(self, tool_name: str) -> Any: ...

    async def get_tool_group(self, toolgroup_id: str) -> Any: ...


@runtime_checkable
@trace_protocol
class ToolRuntimeService(Protocol):
    tool_store: ToolStore | None = None

    rag_tool: RAGToolRuntime | None = None

    # TODO: This needs to be renamed once OPEN API generator name conflict issue is fixed.
    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        """List all tools in the runtime."""
        ...

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        """Run a tool with the given arguments."""
        ...
