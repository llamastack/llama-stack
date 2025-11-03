# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Annotated, Protocol, runtime_checkable

from llama_stack.apis.common.responses import Order, PaginatedResponse
from llama_stack.apis.inference import ToolConfig, ToolResponse, ToolResponseMessage, UserMessage
from llama_stack.core.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import ExtraBodyField

from .models import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
    AgentSessionCreateResponse,
    AgentStepResponse,
    AgentToolGroup,
    AgentTurnResponseStreamChunk,
    Document,
    ResponseGuardrail,
    Session,
    Turn,
)
from .openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseText,
)


@runtime_checkable
@trace_protocol
class AgentsService(Protocol):
    """Agents

    APIs for creating and interacting with agentic systems."""

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        """Create an agent with the given configuration.

        :param agent_config: The configuration for the agent.
        :returns: An AgentCreateResponse with the agent ID.
        """
        ...

    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: list[UserMessage | ToolResponseMessage],
        stream: bool | None = False,
        documents: list[Document] | None = None,
        toolgroups: list[AgentToolGroup] | None = None,
        tool_config: ToolConfig | None = None,
    ) -> Turn | AsyncIterator[AgentTurnResponseStreamChunk]:
        """Create a new turn for an agent.

        :param agent_id: The ID of the agent to create the turn for.
        :param session_id: The ID of the session to create the turn for.
        :param messages: List of messages to start the turn with.
        :param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False.
        :param documents: (Optional) List of documents to create the turn with.
        :param toolgroups: (Optional) List of toolgroups to create the turn with, will be used in addition to the agent's config toolgroups for the request.
        :param tool_config: (Optional) The tool configuration to create the turn with, will be used to override the agent's tool_config.
        :returns: If stream=False, returns a Turn object.
                  If stream=True, returns an SSE event stream of AgentTurnResponseStreamChunk.
        """
        ...

    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: list[ToolResponse],
        stream: bool | None = False,
    ) -> Turn | AsyncIterator[AgentTurnResponseStreamChunk]:
        """Resume an agent turn with executed tool call responses.

        When a Turn has the status `awaiting_input` due to pending input from client side tool calls, this endpoint can be used to submit the outputs from the tool calls once they are ready.

        :param agent_id: The ID of the agent to resume.
        :param session_id: The ID of the session to resume.
        :param turn_id: The ID of the turn to resume.
        :param tool_responses: The tool call responses to resume the turn with.
        :param stream: Whether to stream the response.
        :returns: A Turn object if stream is False, otherwise an AsyncIterator of AgentTurnResponseStreamChunk objects.
        """
        ...

    async def get_agents_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
    ) -> Turn:
        """Retrieve an agent turn by its ID.

        :param agent_id: The ID of the agent to get the turn for.
        :param session_id: The ID of the session to get the turn for.
        :param turn_id: The ID of the turn to get.
        :returns: A Turn.
        """
        ...

    async def get_agents_step(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        step_id: str,
    ) -> AgentStepResponse:
        """Retrieve an agent step by its ID.

        :param agent_id: The ID of the agent to get the step for.
        :param session_id: The ID of the session to get the step for.
        :param turn_id: The ID of the turn to get the step for.
        :param step_id: The ID of the step to get.
        :returns: An AgentStepResponse.
        """
        ...

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        """Create a new session for an agent.

        :param agent_id: The ID of the agent to create the session for.
        :param session_name: The name of the session to create.
        :returns: An AgentSessionCreateResponse.
        """
        ...

    async def get_agents_session(
        self,
        session_id: str,
        agent_id: str,
        turn_ids: list[str] | None = None,
    ) -> Session:
        """Retrieve an agent session by its ID.

        :param session_id: The ID of the session to get.
        :param agent_id: The ID of the agent to get the session for.
        :param turn_ids: (Optional) List of turn IDs to filter the session by.
        :returns: A Session.
        """
        ...

    async def delete_agents_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> None:
        """Delete an agent session by its ID and its associated turns.

        :param session_id: The ID of the session to delete.
        :param agent_id: The ID of the agent to delete the session for.
        """
        ...

    async def delete_agent(
        self,
        agent_id: str,
    ) -> None:
        """Delete an agent by its ID and its associated sessions and turns.

        :param agent_id: The ID of the agent to delete.
        """
        ...

    async def list_agents(self, start_index: int | None = None, limit: int | None = None) -> PaginatedResponse:
        """List all agents.

        :param start_index: The index to start the pagination from.
        :param limit: The number of agents to return.
        :returns: A PaginatedResponse.
        """
        ...

    async def get_agent(self, agent_id: str) -> Agent:
        """Describe an agent by its ID.

        :param agent_id: ID of the agent.
        :returns: An Agent of the agent.
        """
        ...

    async def list_agent_sessions(
        self,
        agent_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        """List all session(s) of a given agent.

        :param agent_id: The ID of the agent to list sessions for.
        :param start_index: The index to start the pagination from.
        :param limit: The number of sessions to return.
        :returns: A PaginatedResponse.
        """
        ...

    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        """Get a model response.

        :param response_id: The ID of the OpenAI response to retrieve.
        :returns: An OpenAIResponseObject.
        """
        ...

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
        guardrails: Annotated[
            list[ResponseGuardrail] | None,
            ExtraBodyField(
                "List of guardrails to apply during response generation. Guardrails provide safety and content moderation."
            ),
        ] = None,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create a model response.

        :param input: Input message(s) to create the response.
        :param model: The underlying LLM used for completions.
        :param prompt: (Optional) Prompt object with ID, version, and variables.
        :param previous_response_id: (Optional) if specified, the new response will be a continuation of the previous response. This can be used to easily fork-off new responses from existing responses.
        :param conversation: (Optional) The ID of a conversation to add the response to. Must begin with 'conv_'. Input and output messages will be automatically added to the conversation.
        :param include: (Optional) Additional fields to include in the response.
        :param guardrails: (Optional) List of guardrails to apply during response generation. Can be guardrail IDs (strings) or guardrail specifications.
        :returns: An OpenAIResponseObject.
        """
        ...

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """List all responses.

        :param after: The ID of the last response to return.
        :param limit: The number of responses to return.
        :param model: The model to filter responses by.
        :param order: The order to sort responses by when sorted by created_at ('asc' or 'desc').
        :returns: A ListOpenAIResponseObject.
        """
        ...

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
        :param order: The order to return the input items in. Default is desc.
        :returns: An ListOpenAIResponseInputItem.
        """
        ...

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        """Delete a response.

        :param response_id: The ID of the OpenAI response to delete.
        :returns: An OpenAIDeleteResponseObject
        """
        ...
