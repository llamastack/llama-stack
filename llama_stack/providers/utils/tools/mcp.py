# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import copy
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, cast

import httpx
from mcp import ClientSession, McpError
from mcp import types as mcp_types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from llama_stack.apis.common.content_types import ImageContentItem, InterleavedContentItem, TextContentItem
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
)
from llama_stack.core.datatypes import AuthenticationRequiredError
from llama_stack.log import get_logger
from llama_stack.providers.utils.tools.ttl_dict import TTLDict

logger = get_logger(__name__, category="tools")

protocol_cache = TTLDict(ttl_seconds=3600)


def resolve_json_schema_refs(schema: Any) -> Any:
    """
    Resolve JSON Schema $ref references using $defs.
    This function takes a JSON schema that may contain $ref and $defs,
    and returns a new schema with all $ref references resolved inline.
    """
    if not isinstance(schema, dict):
        return schema

    # Make a deep copy to avoid modifying the original
    resolved_schema = copy.deepcopy(schema)
    defs = resolved_schema.get("$defs", {})

    def resolve_refs(obj: Any) -> Any:
        """Recursively resolve $ref references in the schema."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                # Extract the reference path
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path[len("#/$defs/") :]
                    if def_name in defs:
                        # Recursively resolve refs in the definition itself
                        resolved_def = resolve_refs(defs[def_name])
                        return resolved_def
                    else:
                        logger.warning(f"Referenced definition '{def_name}' not found in $defs")
                        return obj
                else:
                    logger.warning(f"Unsupported $ref format: {ref_path}")
                    return obj
            else:
                # Recursively process all values in the dict
                return {key: resolve_refs(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process all items in the list
            return [resolve_refs(item) for item in obj]
        else:
            # Return primitive values as-is
            return obj

    # Resolve all refs in the schema
    resolved_schema = resolve_refs(resolved_schema)

    # Remove the $defs section as it's no longer needed
    resolved_schema.pop("$defs", None)

    return resolved_schema


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
                client = sse_client
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
            parameters = []

            # Resolve $ref and $defs in the input schema
            resolved_input_schema = resolve_json_schema_refs(tool.inputSchema)

            for param_name, param_schema in resolved_input_schema.get("properties", {}).items():
                parameters.append(
                    ToolParameter(
                        name=param_name,
                        parameter_type=param_schema.get("type", "string"),
                        description=param_schema.get("description", ""),
                        required="default" not in param_schema,
                        items=param_schema.get("items", None),
                        title=param_schema.get("title", None),
                        default=param_schema.get("default", None),
                    )
                )
            tools.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description,
                    parameters=parameters,
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
                content.append(ImageContentItem(image=item.data))
            elif isinstance(item, mcp_types.EmbeddedResource):
                logger.warning(f"EmbeddedResource is not supported: {item}")
            else:
                raise ValueError(f"Unknown content type: {type(item)}")
        return ToolInvocationResult(
            content=content,
            error_code=1 if result.isError else 0,
        )
