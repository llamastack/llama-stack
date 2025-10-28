# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, cast

import httpx
from mcp import ClientSession, McpError
from mcp import types as mcp_types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from llama_stack.apis.common.content_types import ImageContentItem, InterleavedContentItem, TextContentItem, _URLOrData
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolInvocationResult,
)
from llama_stack.core.datatypes import AuthenticationRequiredError
from llama_stack.log import get_logger
from llama_stack.providers.utils.tools.ttl_dict import TTLDict

logger = get_logger(__name__, category="tools")

protocol_cache = TTLDict(ttl_seconds=3600)


class MCPProtol(Enum):
    UNKNOWN = 0
    STREAMABLE_HTTP = 1
    SSE = 2


@asynccontextmanager
async def client_wrapper(endpoint: str, headers: dict[str, str]) -> AsyncGenerator[ClientSession, Any]:
    # we use a ttl'd dict to cache the happy path protocol for each endpoint
    # but, we always fall back to trying the other protocol if we cannot initialize the session
    connection_strategies = [MCPProtol.STREAMABLE_HTTP, MCPProtol.SSE]
    mcp_protocol = protocol_cache.get(endpoint, default=MCPProtol.UNKNOWN)
    if mcp_protocol == MCPProtol.SSE:
        connection_strategies = [MCPProtol.SSE, MCPProtol.STREAMABLE_HTTP]

    for i, strategy in enumerate(connection_strategies):
        try:
            client = streamablehttp_client
            if strategy == MCPProtol.SSE:
                # sse_client and streamablehttp_client have different signatures, but both
                # are called the same way here, so we cast to Any to avoid type errors
                client = cast(Any, sse_client)
            async with client(endpoint, headers=headers) as client_streams:
                async with ClientSession(read_stream=client_streams[0], write_stream=client_streams[1]) as session:
                    await session.initialize()
                    protocol_cache[endpoint] = strategy
                    yield session
                    return
        except* httpx.HTTPStatusError as eg:
            for exc in eg.exceptions:
                # mypy does not currently narrow the type of `eg.exceptions` based on the `except*` filter,
                # so we explicitly cast each item to httpx.HTTPStatusError. This is safe because
                # `except* httpx.HTTPStatusError` guarantees all exceptions in `eg.exceptions` are of that type.
                err = cast(httpx.HTTPStatusError, exc)
                if err.response.status_code == 401:
                    raise AuthenticationRequiredError(exc) from exc
            if i == len(connection_strategies) - 1:
                raise
        except* httpx.ConnectError as eg:
            # Connection refused, server down, network unreachable
            if i == len(connection_strategies) - 1:
                error_msg = f"Failed to connect to MCP server at {endpoint}: Connection refused"
                logger.error(f"MCP connection error: {error_msg}")
                raise ConnectionError(error_msg) from eg
            else:
                logger.warning(
                    f"failed to connect to MCP server at {endpoint} via {strategy.name}, falling back to {connection_strategies[i + 1].name}"
                )
        except* httpx.TimeoutException as eg:
            # Request timeout, server too slow
            if i == len(connection_strategies) - 1:
                error_msg = f"MCP server at {endpoint} timed out"
                logger.error(f"MCP timeout error: {error_msg}")
                raise TimeoutError(error_msg) from eg
            else:
                logger.warning(
                    f"MCP server at {endpoint} timed out via {strategy.name}, falling back to {connection_strategies[i + 1].name}"
                )
        except* httpx.RequestError as eg:
            # DNS resolution failures, network errors, invalid URLs
            if i == len(connection_strategies) - 1:
                # Get the first exception's message for the error string
                exc_msg = str(eg.exceptions[0]) if eg.exceptions else "Unknown error"
                error_msg = f"Network error connecting to MCP server at {endpoint}: {exc_msg}"
                logger.error(f"MCP network error: {error_msg}")
                raise ConnectionError(error_msg) from eg
            else:
                logger.warning(
                    f"network error connecting to MCP server at {endpoint} via {strategy.name}, falling back to {connection_strategies[i + 1].name}"
                )
        except* McpError:
            if i < len(connection_strategies) - 1:
                logger.warning(
                    f"failed to connect via {strategy.name}, falling back to {connection_strategies[i + 1].name}"
                )
            else:
                raise


async def list_mcp_tools(endpoint: str, headers: dict[str, str]) -> ListToolDefsResponse:
    tools = []
    async with client_wrapper(endpoint, headers) as session:
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tools.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    output_schema=getattr(tool, "outputSchema", None),
                    metadata={
                        "endpoint": endpoint,
                    },
                )
            )
    return ListToolDefsResponse(data=tools)


async def invoke_mcp_tool(
    endpoint: str, headers: dict[str, str], tool_name: str, kwargs: dict[str, Any]
) -> ToolInvocationResult:
    async with client_wrapper(endpoint, headers) as session:
        result = await session.call_tool(tool_name, kwargs)

        content: list[InterleavedContentItem] = []
        for item in result.content:
            if isinstance(item, mcp_types.TextContent):
                content.append(TextContentItem(text=item.text))
            elif isinstance(item, mcp_types.ImageContent):
                content.append(ImageContentItem(image=_URLOrData(data=item.data)))
            elif isinstance(item, mcp_types.EmbeddedResource):
                logger.warning(f"EmbeddedResource is not supported: {item}")
            else:
                raise ValueError(f"Unknown content type: {type(item)}")
        return ToolInvocationResult(
            content=content,
            error_code=1 if result.isError else 0,
        )
