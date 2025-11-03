# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.common.content_types import URL, InterleavedContent
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ToolDef(BaseModel):
    """Tool definition used in runtime contexts."""

    toolgroup_id: str | None = Field(default=None, description="ID of the tool group this tool belongs to")
    name: str = Field(..., description="Name of the tool")
    description: str | None = Field(default=None, description="Human-readable description of what the tool does")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for tool inputs (MCP inputSchema)"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for tool outputs (MCP outputSchema)"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata about the tool")


@json_schema_type
class ToolGroupInput(BaseModel):
    """Input data for registering a tool group."""

    toolgroup_id: str = Field(..., description="Unique identifier for the tool group")
    provider_id: str = Field(..., description="ID of the provider that will handle this tool group")
    args: dict[str, Any] | None = Field(default=None, description="Additional arguments to pass to the provider")
    mcp_endpoint: URL | None = Field(default=None, description="Model Context Protocol endpoint for remote tools")


@json_schema_type
class ToolGroup(Resource):
    """A group of related tools managed together."""

    type: Literal[ResourceType.tool_group] = Field(
        default=ResourceType.tool_group, description="Type of resource, always 'tool_group'"
    )
    mcp_endpoint: URL | None = Field(default=None, description="Model Context Protocol endpoint for remote tools")
    args: dict[str, Any] | None = Field(default=None, description="Additional arguments for the tool group")


@json_schema_type
class ToolInvocationResult(BaseModel):
    """Result of a tool invocation."""

    content: InterleavedContent | None = Field(default=None, description="The output content from the tool execution")
    error_message: str | None = Field(default=None, description="Error message if the tool execution failed")
    error_code: int | None = Field(default=None, description="Numeric error code if the tool execution failed")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata about the tool execution")


class ListToolGroupsResponse(BaseModel):
    """Response containing a list of tool groups."""

    data: list[ToolGroup] = Field(..., description="List of tool groups")


@json_schema_type
class ListToolDefsResponse(BaseModel):
    """Response containing a list of tool definitions."""

    data: list[ToolDef] = Field(..., description="List of tool definitions")


@json_schema_type
class RegisterToolGroupRequest(BaseModel):
    """Request model for registering a tool group."""

    toolgroup_id: str = Field(..., description="The ID of the tool group to register")
    provider_id: str = Field(..., description="The ID of the provider to use for the tool group")
    mcp_endpoint: URL | None = Field(default=None, description="The MCP endpoint to use for the tool group")
    args: dict[str, Any] | None = Field(default=None, description="A dictionary of arguments to pass to the tool group")


@json_schema_type
class InvokeToolRequest(BaseModel):
    """Request model for invoking a tool."""

    tool_name: str = Field(..., description="The name of the tool to invoke")
    kwargs: dict[str, Any] = Field(..., description="A dictionary of arguments to pass to the tool")


class SpecialToolGroup(Enum):
    """Special tool groups with predefined functionality.

    :cvar rag_tool: Retrieval-Augmented Generation tool group for document search and retrieval
    """

    rag_tool = "rag_tool"
