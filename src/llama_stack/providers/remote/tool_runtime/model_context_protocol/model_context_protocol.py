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
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        # this endpoint should be retrieved by getting the tool group right?
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")
        headers, authorization = await self.get_headers_from_request(mcp_endpoint.uri)
        return await list_mcp_tools(endpoint=mcp_endpoint.uri, headers=headers, authorization=authorization)

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        headers, authorization = await self.get_headers_from_request(endpoint)
        return await invoke_mcp_tool(
            endpoint=endpoint,
            tool_name=tool_name,
            kwargs=kwargs,
            headers=headers,
            authorization=authorization,
        )

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> tuple[dict[str, str], str | None]:
        """
        Extract headers and authorization from request provider data.

        For security, Authorization should not be passed via mcp_headers.
        Instead, use a dedicated authorization field in the provider data.

        Returns:
            Tuple of (headers_dict, authorization_token)
            - headers_dict: All headers except Authorization
            - authorization_token: Token from Authorization header (with "Bearer " prefix removed), or None

        Raises:
            ValueError: If Authorization header is found in mcp_headers (security risk)
        """

        def canonicalize_uri(uri: str) -> str:
            return f"{urlparse(uri).netloc or ''}/{urlparse(uri).path or ''}"

        headers = {}
        authorization = None

        provider_data = self.get_request_provider_data()
        if provider_data:
            # Extract headers (excluding Authorization)
            if provider_data.mcp_headers:
                for uri, values in provider_data.mcp_headers.items():
                    if canonicalize_uri(uri) != canonicalize_uri(mcp_endpoint_uri):
                        continue

                    # Security check: reject Authorization header in mcp_headers
                    # This prevents accidentally passing inference tokens to MCP servers
                    for key in values.keys():
                        if key.lower() == "authorization":
                            raise ValueError(
                                "Authorization header cannot be passed via 'mcp_headers'. "
                                "Please use 'mcp_authorization' in provider_data instead."
                            )

                    # Collect all headers (Authorization already rejected above)
                    headers.update(values)

            # Extract authorization from dedicated field
            if provider_data.mcp_authorization:
                canonical_endpoint = canonicalize_uri(mcp_endpoint_uri)
                for uri, token in provider_data.mcp_authorization.items():
                    if canonicalize_uri(uri) == canonical_endpoint:
                        authorization = token
                        break

        return headers, authorization
