# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llama_stack.apis.common.content_types import URL, ContentDelta, InterleavedContent
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
from llama_stack.schema_utils import json_schema_type, register_schema

from .openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponsePrompt,
    OpenAIResponseText,
)


@json_schema_type
class ResponseGuardrailSpec(BaseModel):
    """Specification for a guardrail to apply during response generation."""

    type: str = Field(description="The type/identifier of the guardrail.")
    # TODO: more fields to be added for guardrail configuration


ResponseGuardrail = str | ResponseGuardrailSpec


class Attachment(BaseModel):
    """An attachment to an agent turn."""

    content: InterleavedContent | URL = Field(description="The content of the attachment.")
    mime_type: str = Field(description="The MIME type of the attachment.")


class Document(BaseModel):
    """A document to be used by an agent."""

    content: InterleavedContent | URL = Field(description="The content of the document.")
    mime_type: str = Field(description="The MIME type of the document.")


class StepCommon(BaseModel):
    """A common step in an agent turn."""

    turn_id: str = Field(description="The ID of the turn.")
    step_id: str = Field(description="The ID of the step.")
    started_at: datetime | None = Field(default=None, description="The time the step started.")
    completed_at: datetime | None = Field(default=None, description="The time the step completed.")


class StepType(StrEnum):
    """Type of the step in an agent turn."""

    inference = "inference"
    tool_execution = "tool_execution"
    shield_call = "shield_call"
    memory_retrieval = "memory_retrieval"


@json_schema_type
class InferenceStep(StepCommon):
    """An inference step in an agent turn."""

    model_config = ConfigDict(protected_namespaces=())

    step_type: Literal[StepType.inference] = Field(default=StepType.inference)
    model_response: CompletionMessage = Field(description="The response from the LLM.")


@json_schema_type
class ToolExecutionStep(StepCommon):
    """A tool execution step in an agent turn."""

    step_type: Literal[StepType.tool_execution] = Field(default=StepType.tool_execution)
    tool_calls: list[ToolCall] = Field(description="The tool calls to execute.")
    tool_responses: list[ToolResponse] = Field(description="The tool responses from the tool calls.")


@json_schema_type
class ShieldCallStep(StepCommon):
    """A shield call step in an agent turn."""

    step_type: Literal[StepType.shield_call] = Field(default=StepType.shield_call)
    violation: SafetyViolation | None = Field(default=None, description="The violation from the shield call.")


@json_schema_type
class MemoryRetrievalStep(StepCommon):
    """A memory retrieval step in an agent turn."""

    step_type: Literal[StepType.memory_retrieval] = Field(default=StepType.memory_retrieval)
    # TODO: should this be List[str]?
    vector_store_ids: str = Field(description="The IDs of the vector databases to retrieve context from.")
    inserted_context: InterleavedContent = Field(description="The context retrieved from the vector databases.")


Step = Annotated[
    InferenceStep | ToolExecutionStep | ShieldCallStep | MemoryRetrievalStep,
    Field(discriminator="step_type"),
]


@json_schema_type
class Turn(BaseModel):
    """A single turn in an interaction with an Agentic System."""

    turn_id: str = Field(description="Unique identifier for the turn within a session")
    session_id: str = Field(description="Unique identifier for the conversation session")
    input_messages: list[UserMessage | ToolResponseMessage] = Field(
        description="List of messages that initiated this turn"
    )
    steps: list[Step] = Field(description="Ordered list of processing steps executed during this turn")
    output_message: CompletionMessage = Field(
        description="The model's generated response containing content and metadata"
    )
    output_attachments: list[Attachment] | None = Field(
        default_factory=lambda: [], description="Files or media attached to the agent's response"
    )

    started_at: datetime = Field(description="Timestamp when the turn began")
    completed_at: datetime | None = Field(default=None, description="Timestamp when the turn finished, if completed")


@json_schema_type
class Session(BaseModel):
    """A single session of an interaction with an Agentic System."""

    session_id: str = Field(description="Unique identifier for the conversation session")
    session_name: str = Field(description="Human-readable name for the session")
    turns: list[Turn] = Field(description="List of all turns that have occurred in this session")
    started_at: datetime = Field(description="Timestamp when the session was created")


class AgentToolGroupWithArgs(BaseModel):
    name: str = Field()
    args: dict[str, Any] = Field()


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
    """Configuration for an agent."""

    model: str = Field(description="The model identifier to use for the agent")
    instructions: str = Field(description="The system instructions for the agent")
    name: str | None = Field(
        default=None, description="Optional name for the agent, used in telemetry and identification"
    )
    enable_session_persistence: bool | None = Field(
        default=False, description="Optional flag indicating whether session data has to be persisted"
    )
    response_format: ResponseFormat | None = Field(default=None, description="Optional response format configuration")


@json_schema_type
class Agent(BaseModel):
    """An agent instance with configuration and metadata."""

    agent_id: str = Field(description="Unique identifier for the agent")
    agent_config: AgentConfig = Field(description="Configuration settings for the agent")
    created_at: datetime = Field(description="Timestamp when the agent was created")


class AgentConfigOverridablePerTurn(AgentConfigCommon):
    instructions: str | None = Field(default=None)


class AgentTurnResponseEventType(StrEnum):
    step_start = "step_start"
    step_complete = "step_complete"
    step_progress = "step_progress"

    turn_start = "turn_start"
    turn_complete = "turn_complete"
    turn_awaiting_input = "turn_awaiting_input"


@json_schema_type
class AgentTurnResponseStepStartPayload(BaseModel):
    """Payload for step start events in agent turn responses."""

    event_type: Literal[AgentTurnResponseEventType.step_start] = Field(
        default=AgentTurnResponseEventType.step_start, description="Type of event being reported"
    )
    step_type: StepType = Field(description="Type of step being executed")
    step_id: str = Field(description="Unique identifier for the step within a turn")
    metadata: dict[str, Any] | None = Field(default_factory=lambda: {}, description="Additional metadata for the step")


@json_schema_type
class AgentTurnResponseStepCompletePayload(BaseModel):
    """Payload for step completion events in agent turn responses."""

    event_type: Literal[AgentTurnResponseEventType.step_complete] = Field(
        default=AgentTurnResponseEventType.step_complete, description="Type of event being reported"
    )
    step_type: StepType = Field(description="Type of step being executed")
    step_id: str = Field(description="Unique identifier for the step within a turn")
    step_details: Step = Field(description="Complete details of the executed step")


@json_schema_type
class AgentTurnResponseStepProgressPayload(BaseModel):
    """Payload for step progress events in agent turn responses."""

    model_config = ConfigDict(protected_namespaces=())

    event_type: Literal[AgentTurnResponseEventType.step_progress] = Field(
        default=AgentTurnResponseEventType.step_progress, description="Type of event being reported"
    )
    step_type: StepType = Field(description="Type of step being executed")
    step_id: str = Field(description="Unique identifier for the step within a turn")

    delta: ContentDelta = Field(description="Incremental content changes during step execution")


@json_schema_type
class AgentTurnResponseTurnStartPayload(BaseModel):
    """Payload for turn start events in agent turn responses."""

    event_type: Literal[AgentTurnResponseEventType.turn_start] = Field(
        default=AgentTurnResponseEventType.turn_start, description="Type of event being reported"
    )
    turn_id: str = Field(description="Unique identifier for the turn within a session")


@json_schema_type
class AgentTurnResponseTurnCompletePayload(BaseModel):
    """Payload for turn completion events in agent turn responses."""

    event_type: Literal[AgentTurnResponseEventType.turn_complete] = Field(
        default=AgentTurnResponseEventType.turn_complete, description="Type of event being reported"
    )
    turn: Turn = Field(description="Complete turn data including all steps and results")


@json_schema_type
class AgentTurnResponseTurnAwaitingInputPayload(BaseModel):
    """Payload for turn awaiting input events in agent turn responses."""

    event_type: Literal[AgentTurnResponseEventType.turn_awaiting_input] = Field(
        default=AgentTurnResponseEventType.turn_awaiting_input, description="Type of event being reported"
    )
    turn: Turn = Field(description="Turn data when waiting for external tool responses")


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
    """An event in an agent turn response stream."""

    payload: AgentTurnResponseEventPayload = Field(description="Event-specific payload containing event data")


@json_schema_type
class AgentCreateResponse(BaseModel):
    """Response returned when creating a new agent."""

    agent_id: str = Field(description="Unique identifier for the created agent")


@json_schema_type
class AgentSessionCreateResponse(BaseModel):
    """Response returned when creating a new agent session."""

    session_id: str = Field(description="Unique identifier for the created session")


@json_schema_type
class AgentTurnCreateRequest(AgentConfigOverridablePerTurn):
    """Request to create a new turn for an agent."""

    agent_id: str = Field(description="Unique identifier for the agent")
    session_id: str = Field(description="Unique identifier for the conversation session")

    # TODO: figure out how we can simplify this and make why
    # ToolResponseMessage needs to be here (it is function call
    # execution from outside the system)
    messages: list[UserMessage | ToolResponseMessage] = Field(description="List of messages to start the turn with")

    documents: list[Document] | None = Field(default=None, description="List of documents to provide to the agent")
    toolgroups: list[AgentToolGroup] | None = Field(
        default_factory=lambda: [], description="List of tool groups to make available for this turn"
    )

    stream: bool | None = Field(default=False, description="Whether to stream the response")
    tool_config: ToolConfig | None = Field(default=None, description="Tool configuration to override agent defaults")


@json_schema_type
class AgentTurnResumeRequest(BaseModel):
    """Request to resume an agent turn with tool responses."""

    agent_id: str = Field(description="Unique identifier for the agent")
    session_id: str = Field(description="Unique identifier for the conversation session")
    turn_id: str = Field(description="Unique identifier for the turn within a session")
    tool_responses: list[ToolResponse] = Field(description="List of tool responses to submit to continue the turn")
    stream: bool | None = Field(default=False, description="Whether to stream the response")


@json_schema_type
class AgentTurnResponseStreamChunk(BaseModel):
    """Streamed agent turn completion response."""

    event: AgentTurnResponseEvent = Field(description="Individual event in the agent turn response stream")


@json_schema_type
class AgentStepResponse(BaseModel):
    """Response containing details of a specific agent step."""

    step: Step = Field(description="The complete step data and execution details")


@json_schema_type
class CreateAgentSessionRequest(BaseModel):
    """Request to create a new session for an agent."""

    agent_id: str = Field(..., description="The ID of the agent to create the session for")
    session_name: str = Field(..., description="The name of the session to create")


@json_schema_type
class CreateOpenAIResponseRequest(BaseModel):
    """Request to create a model response."""

    input: str | list[OpenAIResponseInput] = Field(..., description="Input message(s) to create the response")
    model: str = Field(..., description="The underlying LLM used for completions")
    prompt: OpenAIResponsePrompt | None = Field(None, description="Prompt object with ID, version, and variables")
    instructions: str | None = Field(None, description="System instructions")
    previous_response_id: str | None = Field(
        None, description="If specified, the new response will be a continuation of the previous response"
    )
    conversation: str | None = Field(
        None, description="The ID of a conversation to add the response to. Must begin with 'conv_'"
    )
    store: bool = Field(True, description="Whether to store the response")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float | None = Field(None, description="Sampling temperature")
    text: OpenAIResponseText | None = Field(None, description="Text generation parameters")
    tools: list[OpenAIResponseInputTool] | None = Field(None, description="Tools to make available")
    include: list[str] | None = Field(None, description="Additional fields to include in the response")
    max_infer_iters: int = Field(10, description="Maximum number of inference iterations (extension to the OpenAI API)")
    guardrails: list[ResponseGuardrail] | None = Field(
        None, description="List of guardrails to apply during response generation"
    )
