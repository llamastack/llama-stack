# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urlparse

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.datatypes import Api
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolGroup,
    ToolInvocationResult,
    ToolRuntime,
)
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate
from llama_stack.providers.utils.tools.mcp import invoke_mcp_tool, list_mcp_tools

from .config import MCPProviderConfig

logger = get_logger(__name__, category="tools")


class ModelContextProtocolToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    def __init__(self, config: MCPProviderConfig, _deps: dict[Api, Any]):
        self.config = config

    async def initialize(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def list_runtime_tools(
        self,
        tool_group_id: str | None = None,
        mcp_endpoint: URL | None = None,
        authorization: str | None = None,
    ) -> ListToolDefsResponse:
        # this endpoint should be retrieved by getting the tool group right?
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        # Use authorization parameter for MCP servers that require auth
        headers = {}
        return await list_mcp_tools(endpoint=mcp_endpoint.uri, headers=headers, authorization=authorization)

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        # Authorization now comes from request body parameter (not provider-data)
        headers = {}
        return await invoke_mcp_tool(
            endpoint=endpoint,
            tool_name=tool_name,
            kwargs=kwargs,
            headers=headers,
            authorization=authorization,
        )

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> tuple[dict[str, str], str | None]:
        """
        Placeholder method for extracting headers and authorization.

        Note: MCP authentication and headers are now configured via the request body
        (OpenAIResponseInputToolMCP.authorization and .headers fields) and are handled
        by the responses API layer, not at the provider level.

        This method is kept for interface compatibility but returns empty values
        as the tool runtime provider no longer extracts per-request configuration.

        Returns:
            Tuple of (empty_headers_dict, None)
        """
        # Headers and authorization are now handled at the responses API layer
        # via OpenAIResponseInputToolMCP.headers and .authorization fields
        return {}, None
