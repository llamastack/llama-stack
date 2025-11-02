# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

from llama_stack.apis.agents import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
    AgentSessionCreateResponse,
    AgentStepResponse,
    AgentToolGroup,
    Document,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseText,
    Order,
    Session,
    Turn,
)
from llama_stack.apis.agents.agents import ResponseGuardrail
from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.conversations import Conversations
from llama_stack.apis.inference import ToolConfig, ToolResponse, ToolResponseMessage, UserMessage
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.core.datatypes import AccessRule
from llama_stack.log import get_logger

from .config import DanaAgentConfig

logger = get_logger(name=__name__, category="agents::dana")


class DanaAgentsImpl:
    """Dana agent system implementation (stub)."""

    def __init__(
        self,
        config: DanaAgentConfig,
        inference_api: Any,  # Inference
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        conversations_api: Conversations,
        policy: list[AccessRule],
        telemetry_enabled: bool = False,
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api
        self.conversations_api = conversations_api
        self.policy = policy
        self.telemetry_enabled = telemetry_enabled

    async def initialize(self) -> None:
        """Initialize the provider."""
        pass

    async def create_agent(self, agent_config: AgentConfig) -> AgentCreateResponse:
        """Create an agent with the given configuration."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: list[UserMessage | ToolResponseMessage],
        stream: bool | None = False,
        documents: list[Document] | None = None,
        toolgroups: list[AgentToolGroup] | None = None,
        tool_config: ToolConfig | None = None,
    ) -> Turn | AsyncIterator[Any]:
        """Create a new turn for an agent."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: list[ToolResponse],
        stream: bool | None = False,
    ) -> Turn | AsyncIterator[Any]:
        """Resume an agent turn with executed tool call responses."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def get_agents_turn(self, agent_id: str, session_id: str, turn_id: str) -> Turn:
        """Retrieve an agent turn by its ID."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def get_agents_step(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        step_id: str,
    ) -> AgentStepResponse:
        """Retrieve an agent step by its ID."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        """Create a new session for an agent."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def get_agents_session(
        self,
        session_id: str,
        agent_id: str,
        turn_ids: list[str] | None = None,
    ) -> Session:
        """Retrieve an agent session by its ID."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def delete_agents_session(self, session_id: str, agent_id: str) -> None:
        """Delete an agent session by its ID and its associated turns."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def delete_agent(self, agent_id: str) -> None:
        """Delete an agent by its ID and its associated sessions and turns."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def list_agents(self, start_index: int | None = None, limit: int | None = None) -> PaginatedResponse:
        """List all agents."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def get_agent(self, agent_id: str) -> Agent:
        """Describe an agent by its ID."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def list_agent_sessions(
        self,
        agent_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        """List all session(s) of a given agent."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def get_openai_response(self, response_id: str) -> OpenAIResponseObject:
        """Get a model response."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[ResponseGuardrail] | None = None,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create a model response."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """List all responses."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items."""
        raise NotImplementedError("Dana agent implementation is not yet available")

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        """Delete a response."""
        raise NotImplementedError("Dana agent implementation is not yet available")
