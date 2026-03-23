# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from llama_stack_api.common.content_types import URL, InterleavedContent
from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type, webmethod
from llama_stack_api.version import LLAMA_STACK_API_V1


@json_schema_type
class ToolDef(BaseModel):
    """Tool definition used in runtime contexts.

    :param name: Name of the tool
    :param description: (Optional) Human-readable description of what the tool does
    :param input_schema: (Optional) JSON Schema for tool inputs (MCP inputSchema)
    :param output_schema: (Optional) JSON Schema for tool outputs (MCP outputSchema)
    :param metadata: (Optional) Additional metadata about the tool
    :param toolgroup_id: (Optional) ID of the tool group this tool belongs to
    """

    toolgroup_id: str | None = None
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class ToolGroupInput(BaseModel):
    """Input data for registering a tool group.

    :param toolgroup_id: Unique identifier for the tool group
    :param provider_id: ID of the provider that will handle this tool group
    :param args: (Optional) Additional arguments to pass to the provider
    :param mcp_endpoint: (Optional) Model Context Protocol endpoint for remote tools
    """

    toolgroup_id: str
    provider_id: str
    args: dict[str, Any] | None = None
    mcp_endpoint: URL | None = None


@json_schema_type
class ToolGroup(Resource):
    """A group of related tools managed together.

    :param type: Type of resource, always 'tool_group'
    :param mcp_endpoint: (Optional) Model Context Protocol endpoint for remote tools
    :param args: (Optional) Additional arguments for the tool group
    """

    type: Literal[ResourceType.tool_group] = ResourceType.tool_group
    mcp_endpoint: URL | None = None
    args: dict[str, Any] | None = None


@json_schema_type
class ToolInvocationResult(BaseModel):
    """Result of a tool invocation.

    :param content: (Optional) The output content from the tool execution
    :param error_message: (Optional) Error message if the tool execution failed
    :param error_code: (Optional) Numeric error code if the tool execution failed
    :param metadata: (Optional) Additional metadata about the tool execution
    """

    content: InterleavedContent | None = None
    error_message: str | None = None
    error_code: int | None = None
    metadata: dict[str, Any] | None = None


class ToolStore(Protocol):
    async def get_tool(self, tool_name: str) -> ToolDef: ...
    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup: ...


@json_schema_type
class ListToolGroupsResponse(BaseModel):
    """Response containing a list of tool groups.

    :param data: List of tool groups
    """

    data: list[ToolGroup]


@json_schema_type
class ListToolDefsResponse(BaseModel):
    """Response containing a list of tool definitions.

    :param data: List of tool definitions
    """

    data: list[ToolDef]


class ToolGroups(Protocol):
    """Protocol for tool group management and tool discovery.

    Tool groups are auto-registered from configured tool_runtime providers.
    Management methods (register, unregister) are internal. Read-only
    discovery endpoints (list_tools, get_tool) are exposed via HTTP so
    clients can discover which built-in tools are available.
    """

    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool group.

        :param toolgroup_id: The ID of the tool group to register.
        :param provider_id: The ID of the provider to use for the tool group.
        :param mcp_endpoint: The MCP endpoint to use for the tool group.
        :param args: A dictionary of arguments to pass to the tool group.
        """
        ...

    async def get_tool_group(
        self,
        toolgroup_id: str,
    ) -> ToolGroup:
        """Get a tool group by its ID.

        :param toolgroup_id: The ID of the tool group to get.
        :returns: A ToolGroup.
        """
        ...

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        """List tool groups.

        :returns: A ListToolGroupsResponse.
        """
        ...

    @webmethod(route="/tools", method="GET", level=LLAMA_STACK_API_V1)
    async def list_tools(self, toolgroup_id: str | None = None) -> ListToolDefsResponse:
        """List tools with optional tool group filter.

        :param toolgroup_id: The ID of the tool group to list tools for.
        :returns: A ListToolDefsResponse.
        """
        ...

    async def get_tool(
        self,
        tool_name: str,
    ) -> ToolDef:
        """Get a tool by its name.

        :param tool_name: The name of the tool to get.
        :returns: A ToolDef.
        """
        ...

    async def unregister_toolgroup(
        self,
        toolgroup_id: str,
    ) -> None:
        """Unregister a tool group.

        :param toolgroup_id: The ID of the tool group to unregister.
        """
        ...


class SpecialToolGroup(Enum):
    """Special tool groups with predefined functionality.

    :cvar rag_tool: Retrieval-Augmented Generation tool group for document search and retrieval
    """

    rag_tool = "rag_tool"


class ToolRuntime(Protocol):
    """Internal protocol for listing and invoking tools from registered tool groups.

    This protocol is used internally by agents and is not exposed as an HTTP API.
    """

    tool_store: ToolStore | None = None

    async def list_runtime_tools(
        self,
        tool_group_id: str | None = None,
        mcp_endpoint: URL | None = None,
        authorization: str | None = None,
    ) -> ListToolDefsResponse:
        """List all tools in the runtime.

        :param tool_group_id: The ID of the tool group to list tools for.
        :param mcp_endpoint: The MCP endpoint to use for the tool group.
        :param authorization: (Optional) OAuth access token for authenticating with the MCP server.
        :returns: A ListToolDefsResponse.
        """
        ...

    async def invoke_tool(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        authorization: str | None = None,
    ) -> ToolInvocationResult:
        """Run a tool with the given arguments.

        :param tool_name: The name of the tool to invoke.
        :param kwargs: A dictionary of arguments to pass to the tool.
        :param authorization: (Optional) OAuth access token for authenticating with the MCP server.
        :returns: A ToolInvocationResult.
        """
        ...
