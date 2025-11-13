# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urlparse

from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.tools.mcp import invoke_mcp_tool, list_mcp_tools
from llama_stack_api import (
    URL,
    Api,
    ListToolDefsResponse,
    ToolGroup,
    ToolGroupsProtocolPrivate,
    ToolInvocationResult,
    ToolRuntime,
)

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

        # Phase 1: Support both old header-based auth AND new authorization parameter
        # Get headers and auth from provider data (old approach)
        provider_headers, provider_auth = await self.get_headers_from_request(mcp_endpoint.uri)

        # New authorization parameter takes precedence over provider data
        final_authorization = authorization or provider_auth

        return await list_mcp_tools(
            endpoint=mcp_endpoint.uri, headers=provider_headers, authorization=final_authorization
        )

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        # Phase 1: Support both old header-based auth AND new authorization parameter
        # Get headers and auth from provider data (old approach)
        provider_headers, provider_auth = await self.get_headers_from_request(endpoint)

        # New authorization parameter takes precedence over provider data
        final_authorization = authorization or provider_auth

        return await invoke_mcp_tool(
            endpoint=endpoint,
            tool_name=tool_name,
            kwargs=kwargs,
            headers=provider_headers,
            authorization=final_authorization,
        )

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> tuple[dict[str, str], str | None]:
        """
        Extract headers and authorization from request provider data (Phase 1 backward compatibility).

        Phase 1: Temporarily allows Authorization to be passed via mcp_headers for backward compatibility.
        Phase 2: Will enforce that Authorization should use the dedicated authorization parameter instead.

        Returns:
            Tuple of (headers_dict, authorization_token)
            - headers_dict: All headers except Authorization
            - authorization_token: Token from Authorization header (with "Bearer " prefix removed), or None
        """

        def canonicalize_uri(uri: str) -> str:
            return f"{urlparse(uri).netloc or ''}/{urlparse(uri).path or ''}"

        headers = {}
        authorization = None

        provider_data = self.get_request_provider_data()
        if provider_data and hasattr(provider_data, "mcp_headers") and provider_data.mcp_headers:
            for uri, values in provider_data.mcp_headers.items():
                if canonicalize_uri(uri) != canonicalize_uri(mcp_endpoint_uri):
                    continue

                # Phase 1: Extract Authorization from mcp_headers for backward compatibility
                # (Phase 2 will reject this and require the dedicated authorization parameter)
                for key in values.keys():
                    if key.lower() == "authorization":
                        # Extract authorization token and strip "Bearer " prefix if present
                        auth_value = values[key]
                        if auth_value.startswith("Bearer "):
                            authorization = auth_value[7:]  # Remove "Bearer " prefix
                        else:
                            authorization = auth_value
                    else:
                        headers[key] = values[key]

        return headers, authorization
