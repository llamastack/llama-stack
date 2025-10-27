# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from llama_stack.apis.common.content_types import URL, ContentDelta, InterleavedContent
from llama_stack.apis.common.responses import Order, PaginatedResponse
from llama_stack.apis.inference import (
    CompletionMessage,
    ResponseFormat,
    SamplingParams,
    ToolCall,
    ToolChoice,
    ToolConfig,
    ToolPromptFormat,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import SafetyViolation
from llama_stack.apis.tools import ToolDef
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.schema_utils import ExtraBodyField, json_schema_type, register_schema, webmethod

from .openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseText,
)


@json_schema_type
class ResponseGuardrailSpec(BaseModel):
    """Specification for a guardrail to apply during response generation.

    :param type: The type/identifier of the guardrail.
    """

    type: str
    # TODO: more fields to be added for guardrail configuration


ResponseGuardrail = str | ResponseGuardrailSpec


class Attachment(BaseModel):
    """An attachment to an agent turn.

    :param content: The content of the attachment.
    :param mime_type: The MIME type of the attachment.
    """

    content: InterleavedContent | URL
    mime_type: str


class Document(BaseModel):
    """A document to be used by an agent.

    :param content: The content of the document.
    :param mime_type: The MIME type of the document.
    """

    content: InterleavedContent | URL
    mime_type: str


class StepCommon(BaseModel):
    """A common step in an agent turn.

    :param turn_id: The ID of the turn.
    :param step_id: The ID of the step.
    :param started_at: The time the step started.
    :param completed_at: The time the step completed.
    """

    turn_id: str
    step_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None


class StepType(StrEnum):
    """Type of the step in an agent turn.

    :cvar inference: The step is an inference step that calls an LLM.
    :cvar tool_execution: The step is a tool execution step that executes a tool call.
    :cvar shield_call: The step is a shield call step that checks for safety violations.
    :cvar memory_retrieval: The step is a memory retrieval step that retrieves context for vector dbs.
    """

    inference = "inference"
    tool_execution = "tool_execution"
    shield_call = "shield_call"
    memory_retrieval = "memory_retrieval"


@json_schema_type
class InferenceStep(StepCommon):
    """An inference step in an agent turn.

    :param model_response: The response from the LLM.
    """

    model_config = ConfigDict(protected_namespaces=())

    step_type: Literal[StepType.inference] = StepType.inference
    model_response: CompletionMessage


@json_schema_type
class ToolExecutionStep(StepCommon):
    """A tool execution step in an agent turn.

    :param tool_calls: The tool calls to execute.
    :param tool_responses: The tool responses from the tool calls.
    """

    step_type: Literal[StepType.tool_execution] = StepType.tool_execution
    tool_calls: list[ToolCall]
    tool_responses: list[ToolResponse]


@json_schema_type
class ShieldCallStep(StepCommon):
    """A shield call step in an agent turn.

    :param violation: The violation from the shield call.
    """

    step_type: Literal[StepType.shield_call] = StepType.shield_call
    violation: SafetyViolation | None


@json_schema_type
class MemoryRetrievalStep(StepCommon):
    """A memory retrieval step in an agent turn.

    :param vector_db_ids: The IDs of the vector databases to retrieve context from.
    :param inserted_context: The context retrieved from the vector databases.
    """

    step_type: Literal[StepType.memory_retrieval] = StepType.memory_retrieval
    # TODO: should this be List[str]?
    vector_db_ids: str
    inserted_context: InterleavedContent


Step = Annotated[
    InferenceStep | ToolExecutionStep | ShieldCallStep | MemoryRetrievalStep,
    Field(discriminator="step_type"),
]


@json_schema_type
class Turn(BaseModel):
    """A single turn in an interaction with an Agentic System.

    :param turn_id: Unique identifier for the turn within a session
    :param session_id: Unique identifier for the conversation session
    :param input_messages: List of messages that initiated this turn
    :param steps: Ordered list of processing steps executed during this turn
    :param output_message: The model's generated response containing content and metadata
    :param output_attachments: (Optional) Files or media attached to the agent's response
    :param started_at: Timestamp when the turn began
    :param completed_at: (Optional) Timestamp when the turn finished, if completed
    """

    turn_id: str
    session_id: str
    input_messages: list[UserMessage | ToolResponseMessage]
    steps: list[Step]
    output_message: CompletionMessage
    output_attachments: list[Attachment] | None = Field(default_factory=lambda: [])

    started_at: datetime
    completed_at: datetime | None = None


@json_schema_type
class Session(BaseModel):
    """A single session of an interaction with an Agentic System.

    :param session_id: Unique identifier for the conversation session
    :param session_name: Human-readable name for the session
    :param turns: List of all turns that have occurred in this session
    :param started_at: Timestamp when the session was created
    """

    session_id: str
    session_name: str
    turns: list[Turn]
    started_at: datetime


class AgentToolGroupWithArgs(BaseModel):
    name: str
    args: dict[str, Any]


AgentToolGroup = str | AgentToolGroupWithArgs
register_schema(AgentToolGroup, name="AgentTool")


class AgentConfigCommon(BaseModel):
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)

    input_shields: list[str] | None = Field(default_factory=lambda: [])
    output_shields: list[str] | None = Field(default_factory=lambda: [])
    toolgroups: list[AgentToolGroup] | None = Field(default_factory=lambda: [])
    client_tools: list[ToolDef] | None = Field(default_factory=lambda: [])
    tool_choice: ToolChoice | None = Field(default=None, deprecated="use tool_config instead")
    tool_prompt_format: ToolPromptFormat | None = Field(default=None, deprecated="use tool_config instead")
    tool_config: ToolConfig | None = Field(default=None)

    max_infer_iters: int | None = 10

    def model_post_init(self, __context):
        if self.tool_config:
            if self.tool_choice and self.tool_config.tool_choice != self.tool_choice:
                raise ValueError("tool_choice is deprecated. Use tool_choice in tool_config instead.")
            if self.tool_prompt_format and self.tool_config.tool_prompt_format != self.tool_prompt_format:
                raise ValueError("tool_prompt_format is deprecated. Use tool_prompt_format in tool_config instead.")
        else:
            params = {}
            if self.tool_choice:
                params["tool_choice"] = self.tool_choice
            if self.tool_prompt_format:
                params["tool_prompt_format"] = self.tool_prompt_format
            self.tool_config = ToolConfig(**params)


@json_schema_type
class AgentConfig(AgentConfigCommon):
    """Configuration for an agent.

    :param model: The model identifier to use for the agent
    :param instructions: The system instructions for the agent
    :param name: Optional name for the agent, used in telemetry and identification
    :param enable_session_persistence: Optional flag indicating whether session data has to be persisted
    :param response_format: Optional response format configuration
    """

    model: str
    instructions: str
    name: str | None = None
    enable_session_persistence: bool | None = False
    response_format: ResponseFormat | None = None


@json_schema_type
class Agent(BaseModel):
    """An agent instance with configuration and metadata.

    :param agent_id: Unique identifier for the agent
    :param agent_config: Configuration settings for the agent
    :param created_at: Timestamp when the agent was created
    """

    agent_id: str
    agent_config: AgentConfig
    created_at: datetime


class AgentConfigOverridablePerTurn(AgentConfigCommon):
    instructions: str | None = None


class AgentTurnResponseEventType(StrEnum):
    step_start = "step_start"
    step_complete = "step_complete"
    step_progress = "step_progress"

    turn_start = "turn_start"
    turn_complete = "turn_complete"
    turn_awaiting_input = "turn_awaiting_input"


@json_schema_type
class AgentTurnResponseStepStartPayload(BaseModel):
    """Payload for step start events in agent turn responses.

    :param event_type: Type of event being reported
    :param step_type: Type of step being executed
    :param step_id: Unique identifier for the step within a turn
    :param metadata: (Optional) Additional metadata for the step
    """

    event_type: Literal[AgentTurnResponseEventType.step_start] = AgentTurnResponseEventType.step_start
    step_type: StepType
    step_id: str
    metadata: dict[str, Any] | None = Field(default_factory=lambda: {})


@json_schema_type
class AgentTurnResponseStepCompletePayload(BaseModel):
    """Payload for step completion events in agent turn responses.

    :param event_type: Type of event being reported
    :param step_type: Type of step being executed
    :param step_id: Unique identifier for the step within a turn
    :param step_details: Complete details of the executed step
    """

    event_type: Literal[AgentTurnResponseEventType.step_complete] = AgentTurnResponseEventType.step_complete
    step_type: StepType
    step_id: str
    step_details: Step


@json_schema_type
class AgentTurnResponseStepProgressPayload(BaseModel):
    """Payload for step progress events in agent turn responses.

    :param event_type: Type of event being reported
    :param step_type: Type of step being executed
    :param step_id: Unique identifier for the step within a turn
    :param delta: Incremental content changes during step execution
    """

    model_config = ConfigDict(protected_namespaces=())

    event_type: Literal[AgentTurnResponseEventType.step_progress] = AgentTurnResponseEventType.step_progress
    step_type: StepType
    step_id: str

    delta: ContentDelta


@json_schema_type
class AgentTurnResponseTurnStartPayload(BaseModel):
    """Payload for turn start events in agent turn responses.

    :param event_type: Type of event being reported
    :param turn_id: Unique identifier for the turn within a session
    """

    event_type: Literal[AgentTurnResponseEventType.turn_start] = AgentTurnResponseEventType.turn_start
    turn_id: str


@json_schema_type
class AgentTurnResponseTurnCompletePayload(BaseModel):
    """Payload for turn completion events in agent turn responses.

    :param event_type: Type of event being reported
    :param turn: Complete turn data including all steps and results
    """

    event_type: Literal[AgentTurnResponseEventType.turn_complete] = AgentTurnResponseEventType.turn_complete
    turn: Turn


@json_schema_type
class AgentTurnResponseTurnAwaitingInputPayload(BaseModel):
    """Payload for turn awaiting input events in agent turn responses.

    :param event_type: Type of event being reported
    :param turn: Turn data when waiting for external tool responses
    """

    event_type: Literal[AgentTurnResponseEventType.turn_awaiting_input] = AgentTurnResponseEventType.turn_awaiting_input
    turn: Turn


AgentTurnResponseEventPayload = Annotated[
    AgentTurnResponseStepStartPayload
    | AgentTurnResponseStepProgressPayload
    | AgentTurnResponseStepCompletePayload
    | AgentTurnResponseTurnStartPayload
    | AgentTurnResponseTurnCompletePayload
    | AgentTurnResponseTurnAwaitingInputPayload,
    Field(discriminator="event_type"),
]
register_schema(AgentTurnResponseEventPayload, name="AgentTurnResponseEventPayload")


@json_schema_type
class AgentTurnResponseEvent(BaseModel):
    """An event in an agent turn response stream.

    :param payload: Event-specific payload containing event data
    """

    payload: AgentTurnResponseEventPayload


@json_schema_type
class AgentCreateResponse(BaseModel):
    """Response returned when creating a new agent.

    :param agent_id: Unique identifier for the created agent
    """

    agent_id: str


@json_schema_type
class AgentSessionCreateResponse(BaseModel):
    """Response returned when creating a new agent session.

    :param session_id: Unique identifier for the created session
    """

    session_id: str


@json_schema_type
class AgentTurnCreateRequest(AgentConfigOverridablePerTurn):
    """Request to create a new turn for an agent.

    :param agent_id: Unique identifier for the agent
    :param session_id: Unique identifier for the conversation session
    :param messages: List of messages to start the turn with
    :param documents: (Optional) List of documents to provide to the agent
    :param toolgroups: (Optional) List of tool groups to make available for this turn
    :param stream: (Optional) Whether to stream the response
    :param tool_config: (Optional) Tool configuration to override agent defaults
    """

    agent_id: str
    session_id: str

    # TODO: figure out how we can simplify this and make why
    # ToolResponseMessage needs to be here (it is function call
    # execution from outside the system)
    messages: list[UserMessage | ToolResponseMessage]

    documents: list[Document] | None = None
    toolgroups: list[AgentToolGroup] | None = Field(default_factory=lambda: [])

    stream: bool | None = False
    tool_config: ToolConfig | None = None


@json_schema_type
class AgentTurnResumeRequest(BaseModel):
    """Request to resume an agent turn with tool responses.

    :param agent_id: Unique identifier for the agent
    :param session_id: Unique identifier for the conversation session
    :param turn_id: Unique identifier for the turn within a session
    :param tool_responses: List of tool responses to submit to continue the turn
    :param stream: (Optional) Whether to stream the response
    """

    agent_id: str
    session_id: str
    turn_id: str
    tool_responses: list[ToolResponse]
    stream: bool | None = False


@json_schema_type
class AgentTurnResponseStreamChunk(BaseModel):
    """Streamed agent turn completion response.

    :param event: Individual event in the agent turn response stream
    """

    event: AgentTurnResponseEvent


@json_schema_type
class AgentStepResponse(BaseModel):
    """Response containing details of a specific agent step.

    :param step: The complete step data and execution details
    """

    step: Step


@runtime_checkable
class Agents(Protocol):
    """Agents

    APIs for creating and interacting with agentic systems."""

    @webmethod(
        route="/agents",
        method="POST",
        descriptive_name="create_agent",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents",
        method="POST",
        descriptive_name="create_agent",
        level=LLAMA_STACK_API_V1ALPHA,
    )
    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        """Create an agent with the given configuration.

        :param agent_config: The configuration for the agent.
        :returns: An AgentCreateResponse with the agent ID.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn",
        method="POST",
        descriptive_name="create_agent_turn",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn",
        method="POST",
        descriptive_name="create_agent_turn",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/resume",
        method="POST",
        descriptive_name="resume_agent_turn",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/resume",
        method="POST",
        descriptive_name="resume_agent_turn",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}",
        method="GET",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}",
        method="GET",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}",
        method="GET",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}",
        method="GET",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}/session",
        method="POST",
        descriptive_name="create_agent_session",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session",
        method="POST",
        descriptive_name="create_agent_session",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}",
        method="GET",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session/{session_id}",
        method="GET",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}",
        method="DELETE",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(
        route="/agents/{agent_id}/session/{session_id}",
        method="DELETE",
        level=LLAMA_STACK_API_V1ALPHA,
    )
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

    @webmethod(
        route="/agents/{agent_id}",
        method="DELETE",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(route="/agents/{agent_id}", method="DELETE", level=LLAMA_STACK_API_V1ALPHA)
    async def delete_agent(
        self,
        agent_id: str,
    ) -> None:
        """Delete an agent by its ID and its associated sessions and turns.

        :param agent_id: The ID of the agent to delete.
        """
        ...

    @webmethod(route="/agents", method="GET", deprecated=True, level=LLAMA_STACK_API_V1)
    @webmethod(route="/agents", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_agents(self, start_index: int | None = None, limit: int | None = None) -> PaginatedResponse:
        """List all agents.

        :param start_index: The index to start the pagination from.
        :param limit: The number of agents to return.
        :returns: A PaginatedResponse.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}",
        method="GET",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(route="/agents/{agent_id}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_agent(self, agent_id: str) -> Agent:
        """Describe an agent by its ID.

        :param agent_id: ID of the agent.
        :returns: An Agent of the agent.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/sessions",
        method="GET",
        deprecated=True,
        level=LLAMA_STACK_API_V1,
    )
    @webmethod(route="/agents/{agent_id}/sessions", method="GET", level=LLAMA_STACK_API_V1ALPHA)
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

    # We situate the OpenAI Responses API in the Agents API just like we did things
    # for Inference. The Responses API, in its intent, serves the same purpose as
    # the Agents API above -- it is essentially a lightweight "agentic loop" with
    # integrated tool calling.
    #
    # Both of these APIs are inherently stateful.

    @webmethod(
        route="/openai/v1/responses/{response_id}",
        method="GET",
        level=LLAMA_STACK_API_V1,
        deprecated=True,
    )
    @webmethod(route="/responses/{response_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        """Get a model response.

        :param response_id: The ID of the OpenAI response to retrieve.
        :returns: An OpenAIResponseObject.
        """
        ...

    @webmethod(route="/openai/v1/responses", method="POST", level=LLAMA_STACK_API_V1, deprecated=True)
    @webmethod(route="/responses", method="POST", level=LLAMA_STACK_API_V1)
    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,  # this is an extension to the OpenAI API
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
        :param previous_response_id: (Optional) if specified, the new response will be a continuation of the previous response. This can be used to easily fork-off new responses from existing responses.
        :param conversation: (Optional) The ID of a conversation to add the response to. Must begin with 'conv_'. Input and output messages will be automatically added to the conversation.
        :param include: (Optional) Additional fields to include in the response.
        :param guardrails: (Optional) List of guardrails to apply during response generation. Can be guardrail IDs (strings) or guardrail specifications.
        :returns: An OpenAIResponseObject.
        """
        ...

    @webmethod(route="/openai/v1/responses", method="GET", level=LLAMA_STACK_API_V1, deprecated=True)
    @webmethod(route="/responses", method="GET", level=LLAMA_STACK_API_V1)
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

    @webmethod(
        route="/openai/v1/responses/{response_id}/input_items", method="GET", level=LLAMA_STACK_API_V1, deprecated=True
    )
    @webmethod(route="/responses/{response_id}/input_items", method="GET", level=LLAMA_STACK_API_V1)
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

    @webmethod(route="/openai/v1/responses/{response_id}", method="DELETE", level=LLAMA_STACK_API_V1, deprecated=True)
    @webmethod(route="/responses/{response_id}", method="DELETE", level=LLAMA_STACK_API_V1)
    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        """Delete a response.

        :param response_id: The ID of the OpenAI response to delete.
        :returns: An OpenAIDeleteResponseObject
        """
        ...
