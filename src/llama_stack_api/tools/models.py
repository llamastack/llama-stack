# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Tools API requests and responses.

This module defines the request and response models for the Tools API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from llama_stack_api.common.content_types import URL, InterleavedContent
from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class ToolDef(BaseModel):
    """Tool definition used in runtime contexts."""

    toolgroup_id: str | None = None
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class ToolGroupInput(BaseModel):
    """Input data for registering a tool group."""

    toolgroup_id: str
    provider_id: str
    args: dict[str, Any] | None = None
    mcp_endpoint: URL | None = None


@json_schema_type
class ToolGroup(Resource):
    """A group of related tools managed together."""

    type: Literal[ResourceType.tool_group] = ResourceType.tool_group
    mcp_endpoint: URL | None = None
    args: dict[str, Any] | None = None


@json_schema_type
class ToolInvocationResult(BaseModel):
    """Result of a tool invocation."""

    content: InterleavedContent | None = None
    error_message: str | None = None
    error_code: int | None = None
    metadata: dict[str, Any] | None = None


class ToolStore(Protocol):
    async def get_tool(self, tool_name: str) -> ToolDef: ...
    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup: ...


@json_schema_type
class ListToolGroupsResponse(BaseModel):
    """Response containing a list of tool groups."""

    data: list[ToolGroup]


@json_schema_type
class ListToolDefsResponse(BaseModel):
    """Response containing a list of tool definitions."""

    data: list[ToolDef]


@json_schema_type
class ListToolsRequest(BaseModel):
    """Request model for listing tools."""

    toolgroup_id: str | None = Field(default=None, description="The ID of the tool group to filter tools by.")


class SpecialToolGroup(Enum):
    """Special tool groups with predefined functionality.

    :cvar rag_tool: Retrieval-Augmented Generation tool group for document search and retrieval
    """

    rag_tool = "rag_tool"


__all__ = [
    "ListToolDefsResponse",
    "ListToolGroupsResponse",
    "ListToolsRequest",
    "SpecialToolGroup",
    "ToolDef",
    "ToolGroup",
    "ToolGroupInput",
    "ToolInvocationResult",
    "ToolStore",
]
