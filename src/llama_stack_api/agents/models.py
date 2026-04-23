# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the Anthropic Agents API.

These models define the request and response shapes for the /v1alpha/agents endpoint,
following the Anthropic Managed Agents API specification.

Anthropic API Reference: https://docs.anthropic.com/en/api/beta/agents
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Anthropic API version we target
ANTHROPIC_AGENTS_VERSION = "managed-agents-2026-04-01"

# -- Permission policies --


class AlwaysAllowPolicy(BaseModel):
    """Tool calls are automatically approved without user confirmation."""

    type: Literal["always_allow"] = "always_allow"


class AlwaysAskPolicy(BaseModel):
    """Tool calls require user confirmation before execution."""

    type: Literal["always_ask"] = "always_ask"


PermissionPolicy = Annotated[
    AlwaysAllowPolicy | AlwaysAskPolicy,
    Field(discriminator="type"),
]


# -- MCP Server definitions --


class MCPServerURL(BaseModel):
    """MCP server definition with URL endpoint."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique name for this server.")
    type: Literal["url"] = "url"
    url: str = Field(..., description="Endpoint URL for the MCP server.")


# -- Tool definitions --


class AgentToolConfig(BaseModel):
    """Configuration for a single built-in agent tool."""

    name: Literal["bash", "edit", "read", "write", "glob", "grep", "web_fetch", "web_search"] = Field(
        ..., description="Built-in agent tool identifier."
    )
    enabled: bool | None = Field(None, description="Whether this tool is enabled. Overrides default_config.")
    permission_policy: PermissionPolicy | None = Field(None, description="Permission policy for tool execution.")


class AgentToolsetDefaultConfig(BaseModel):
    """Default configuration for all tools in a toolset."""

    enabled: bool | None = Field(None, description="Whether tools are enabled by default. Defaults to true.")
    permission_policy: PermissionPolicy | None = Field(None, description="Default permission policy.")


class AgentToolset20260401(BaseModel):
    """Configuration for built-in agent tools (bash, edit, read, write, glob, grep, web_fetch, web_search)."""

    type: Literal["agent_toolset_20260401"] = "agent_toolset_20260401"
    configs: list[AgentToolConfig] | None = Field(None, description="Per-tool configuration overrides.")
    default_config: AgentToolsetDefaultConfig | None = Field(None, description="Default config for all tools.")


class MCPToolConfig(BaseModel):
    """Configuration for a single MCP tool."""

    name: str = Field(..., min_length=1, max_length=128, description="Name of the MCP tool to configure.")
    enabled: bool | None = Field(None, description="Whether this tool is enabled. Overrides default_config.")
    permission_policy: PermissionPolicy | None = Field(None, description="Permission policy for tool execution.")


class MCPToolsetDefaultConfig(BaseModel):
    """Default configuration for all tools from an MCP server."""

    enabled: bool | None = Field(None, description="Whether tools are enabled by default. Defaults to true.")
    permission_policy: PermissionPolicy | None = Field(None, description="Default permission policy.")


class MCPToolset(BaseModel):
    """Configuration for tools from an MCP server."""

    type: Literal["mcp_toolset"] = "mcp_toolset"
    mcp_server_name: str = Field(
        ..., min_length=1, max_length=255, description="Name of the MCP server from mcp_servers array."
    )
    configs: list[MCPToolConfig] | None = Field(None, description="Per-tool configuration overrides.")
    default_config: MCPToolsetDefaultConfig | None = Field(None, description="Default config for all MCP tools.")


class CustomToolInputSchema(BaseModel):
    """JSON Schema for custom tool input parameters."""

    type: Literal["object"] | None = Field(None, description="Must be 'object' for tool input schemas.")
    properties: dict[str, Any] | None = Field(None, description="JSON Schema properties for input parameters.")
    required: list[str] | None = Field(None, description="List of required property names.")


class CustomTool(BaseModel):
    """A custom tool that is executed by the API client rather than the agent."""

    type: Literal["custom"] = "custom"
    name: str = Field(..., min_length=1, max_length=128, description="Unique name for the tool.")
    description: str = Field(..., min_length=1, max_length=1024, description="Description of what the tool does.")
    input_schema: CustomToolInputSchema = Field(..., description="JSON Schema for tool input parameters.")


AgentTool = Annotated[
    AgentToolset20260401 | MCPToolset | CustomTool,
    Field(discriminator="type"),
]


# -- Skill definitions --


class AnthropicSkill(BaseModel):
    """An Anthropic-managed skill."""

    type: Literal["anthropic"] = "anthropic"
    skill_id: str = Field(..., description="Identifier of the Anthropic skill (e.g., 'xlsx').")
    version: str | None = Field(None, description="Version to pin. Defaults to latest if omitted.")


class CustomSkill(BaseModel):
    """A user-created custom skill."""

    type: Literal["custom"] = "custom"
    skill_id: str = Field(..., description="Tagged ID of the custom skill (e.g., 'skill_01XJ5...').")
    version: str | None = Field(None, description="Version to pin. Defaults to latest if omitted.")


AgentSkill = Annotated[
    AnthropicSkill | CustomSkill,
    Field(discriminator="type"),
]


# -- Agent configuration --


class CreateAgentRequest(BaseModel):
    """Request body for POST /v1alpha/agents."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model identifier (e.g., 'claude-sonnet-4-6', 'llama-3.3-70b').")
    name: str = Field(..., min_length=1, max_length=256, description="Human-readable name for the agent.")
    description: str | None = Field(None, max_length=2048, description="Description of what the agent does.")
    system: str | None = Field(None, max_length=100000, description="System prompt for the agent (max 100KB).")
    mcp_servers: list[MCPServerURL] | None = Field(
        None, max_length=20, description="MCP servers this agent connects to (max 20)."
    )
    tools: list[AgentTool] | None = Field(
        None, max_length=128, description="Tool configurations available to the agent (max 128 tools total)."
    )
    skills: list[AgentSkill] | None = Field(None, max_length=20, description="Skills available to the agent (max 20).")
    metadata: dict[str, str] | None = Field(
        None, description="User-defined metadata (max 16 pairs, keys ≤64 chars, values ≤512 chars)."
    )


class UpdateAgentRequest(BaseModel):
    """Request body for POST /v1alpha/agents/{agent_id} (update/versioning)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=256)
    description: str | None = Field(None, max_length=2048)
    system: str | None = Field(None, max_length=100000)
    mcp_servers: list[MCPServerURL] | None = Field(None, max_length=20)
    tools: list[AgentTool] | None = Field(None, max_length=128)
    skills: list[AgentSkill] | None = Field(None, max_length=20)
    metadata: dict[str, str] | None = None


class AgentObject(BaseModel):
    """Agent configuration object returned from the API."""

    id: str = Field(..., description="Unique agent identifier (agent_ prefix).")
    type: Literal["agent"] = "agent"
    created_at: str = Field(..., description="RFC 3339 timestamp when agent was created.")
    updated_at: str = Field(..., description="RFC 3339 timestamp when agent was last updated.")
    archived_at: str | None = Field(None, description="RFC 3339 timestamp when agent was archived, or null.")
    version: int = Field(..., description="Agent version number (auto-incremented on update).")
    model: str = Field(..., description="Model identifier.")
    name: str = Field(..., description="Human-readable name for the agent.")
    description: str = Field(..., description="Description of what the agent does.")
    system: str = Field(..., description="System prompt for the agent.")
    mcp_servers: list[MCPServerURL] = Field(..., description="MCP servers this agent connects to.")
    tools: list[AgentTool] = Field(..., description="Tool configurations available to the agent.")
    skills: list[AgentSkill] = Field(..., description="Skills available to the agent.")
    metadata: dict[str, str] = Field(..., description="User-defined metadata.")


class ListAgentsRequest(BaseModel):
    """Request parameters for GET /v1alpha/agents."""

    limit: int | None = Field(None, ge=1, le=100, description="Max number of agents to return.")
    after: str | None = Field(None, description="Cursor for pagination (agent ID).")
    archived: bool | None = Field(None, description="Filter by archived status.")


class ListAgentsResponse(BaseModel):
    """Response from GET /v1alpha/agents."""

    object: Literal["list"] = "list"
    data: list[AgentObject]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


class ArchiveAgentRequest(BaseModel):
    """Request body for POST /v1alpha/agents/{agent_id}/archive."""

    # Empty body for now, may add options later
    pass


# -- Error response --


class _AnthropicErrorDetail(BaseModel):
    type: str
    message: str


class AnthropicAgentsErrorResponse(BaseModel):
    """Anthropic-format error response for Agents API."""

    type: Literal["error"] = "error"
    error: _AnthropicErrorDetail
