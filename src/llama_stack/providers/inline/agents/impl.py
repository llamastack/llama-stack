# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Anthropic Agents API implementation.

Provides agent configuration management with in-memory storage.
Future versions will support persistent storage backends.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from llama_stack.log import get_logger
from llama_stack_api import AgentNotFoundError
from llama_stack_api.agents import (
    AgentObject,
    Agents,
    ArchiveAgentRequest,
    CreateAgentRequest,
    ListAgentsRequest,
    ListAgentsResponse,
    UpdateAgentRequest,
)

from .config import AgentsConfig

logger = get_logger(name=__name__, category="agents")


def _rfc3339_timestamp() -> str:
    """Generate RFC 3339 timestamp string."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class BuiltinAgentsImpl(Agents):
    """Built-in implementation of the Anthropic Agents API."""

    def __init__(self, config: AgentsConfig):
        self.config = config
        # In-memory storage: agent_id -> AgentObject
        self._agents: dict[str, AgentObject] = {}
        # Track all versions: agent_id -> list of AgentObjects
        self._agent_versions: dict[str, list[AgentObject]] = {}

    async def initialize(self) -> None:
        """Initialize the agents provider."""
        logger.info("Initializing built-in Agents provider")

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        pass

    async def create_agent(self, request: CreateAgentRequest) -> AgentObject:
        """Create a new agent configuration."""
        agent_id = f"agent_{uuid.uuid4().hex[:24]}"
        timestamp = _rfc3339_timestamp()

        agent = AgentObject(
            id=agent_id,
            created_at=timestamp,
            updated_at=timestamp,
            archived_at=None,
            version=1,
            model=request.model,
            name=request.name,
            description=request.description or "",
            system=request.system or "",
            mcp_servers=request.mcp_servers or [],
            tools=request.tools or [],
            skills=request.skills or [],
            metadata=request.metadata or {},
        )

        self._agents[agent_id] = agent
        self._agent_versions[agent_id] = [agent]

        logger.info("Created agent", agent_id=agent_id, name=request.name, model=request.model)
        return agent

    async def list_agents(
        self,
        request: ListAgentsRequest | None = None,
    ) -> ListAgentsResponse:
        """List all agent configurations with optional filtering and pagination."""
        request = request or ListAgentsRequest()

        # Apply archived filter
        agents = list(self._agents.values())
        if request.archived is not None:
            is_archived = [a for a in agents if a.archived_at is not None]
            is_not_archived = [a for a in agents if a.archived_at is None]
            agents = is_archived if request.archived else is_not_archived

        # Sort by creation time (newest first)
        agents.sort(key=lambda a: a.created_at, reverse=True)

        # Apply pagination
        if request.after:
            # Find the index of the 'after' agent
            try:
                after_idx = next(i for i, a in enumerate(agents) if a.id == request.after)
                agents = agents[after_idx + 1 :]
            except StopIteration:
                # 'after' agent not found, return empty list
                agents = []

        # Apply limit
        limit = request.limit or 100
        has_more = len(agents) > limit
        agents = agents[:limit]

        first_id = agents[0].id if agents else None
        last_id = agents[-1].id if agents else None

        return ListAgentsResponse(
            data=agents,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def get_agent(self, agent_id: str) -> AgentObject:
        """Retrieve an agent configuration by ID."""
        agent = self._agents.get(agent_id)
        if not agent:
            raise AgentNotFoundError(agent_id)

        return agent

    async def update_agent(
        self,
        agent_id: str,
        request: UpdateAgentRequest,
    ) -> AgentObject:
        """Update an agent configuration (creates a new version)."""
        current_agent = self._agents.get(agent_id)
        if not current_agent:
            raise AgentNotFoundError(agent_id)

        # Cannot update archived agents
        if current_agent.archived_at is not None:
            raise ValueError(f"Cannot update archived agent '{agent_id}'")

        # Create new version with updated fields
        new_version = current_agent.version + 1
        timestamp = _rfc3339_timestamp()

        updated_agent = AgentObject(
            id=agent_id,
            created_at=current_agent.created_at,  # Keep original creation time
            updated_at=timestamp,
            archived_at=None,
            version=new_version,
            model=current_agent.model,  # Model cannot be changed
            name=request.name if request.name is not None else current_agent.name,
            description=request.description if request.description is not None else current_agent.description,
            system=request.system if request.system is not None else current_agent.system,
            mcp_servers=request.mcp_servers if request.mcp_servers is not None else current_agent.mcp_servers,
            tools=request.tools if request.tools is not None else current_agent.tools,
            skills=request.skills if request.skills is not None else current_agent.skills,
            metadata=request.metadata if request.metadata is not None else current_agent.metadata,
        )

        # Store the new version
        self._agents[agent_id] = updated_agent
        self._agent_versions[agent_id].append(updated_agent)

        logger.info("Updated agent", agent_id=agent_id, version=new_version)
        return updated_agent

    async def archive_agent(
        self,
        agent_id: str,
        request: ArchiveAgentRequest | None = None,
    ) -> AgentObject:
        """Archive an agent configuration."""
        agent = self._agents.get(agent_id)
        if not agent:
            raise AgentNotFoundError(agent_id)

        if agent.archived_at is not None:
            # Already archived, return as-is
            return agent

        # Create archived copy
        timestamp = _rfc3339_timestamp()
        archived_agent = AgentObject(
            id=agent.id,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            archived_at=timestamp,
            version=agent.version,
            model=agent.model,
            name=agent.name,
            description=agent.description,
            system=agent.system,
            mcp_servers=agent.mcp_servers,
            tools=agent.tools,
            skills=agent.skills,
            metadata=agent.metadata,
        )

        self._agents[agent_id] = archived_agent

        logger.info("Archived agent", agent_id=agent_id)
        return archived_agent
