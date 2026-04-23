# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    AgentObject,
    ArchiveAgentRequest,
    CreateAgentRequest,
    ListAgentsRequest,
    ListAgentsResponse,
    UpdateAgentRequest,
)


@runtime_checkable
class Agents(Protocol):
    """Protocol for the Anthropic Agents API."""

    async def create_agent(
        self,
        request: CreateAgentRequest,
    ) -> AgentObject: ...

    async def list_agents(
        self,
        request: ListAgentsRequest | None = None,
    ) -> ListAgentsResponse: ...

    async def get_agent(
        self,
        agent_id: str,
    ) -> AgentObject: ...

    async def update_agent(
        self,
        agent_id: str,
        request: UpdateAgentRequest,
    ) -> AgentObject: ...

    async def archive_agent(
        self,
        agent_id: str,
        request: ArchiveAgentRequest | None = None,
    ) -> AgentObject: ...
