# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import copy
import json
import re
import uuid
import warnings
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import httpx

from llama_stack.apis.agents import (
    AgentConfig,
    AgentToolGroup,
    AgentToolGroupWithArgs,
    AgentTurnCreateRequest,
    AgentTurnResponseEvent,
    AgentTurnResponseEventType,
    AgentTurnResponseStepCompletePayload,
    AgentTurnResponseStepProgressPayload,
    AgentTurnResponseStepStartPayload,
    AgentTurnResponseStreamChunk,
    AgentTurnResponseTurnAwaitingInputPayload,
    AgentTurnResponseTurnCompletePayload,
    AgentTurnResumeRequest,
    Attachment,
    Document,
    InferenceStep,
    ShieldCallStep,
    StepType,
    ToolExecutionStep,
    Turn,
)
from llama_stack.apis.common.content_types import (
    URL,
    TextContentItem,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.common.errors import SessionNotFoundError
from llama_stack.apis.inference import (
    ChatCompletionResponseEventType,
    CompletionMessage,
    Inference,
    Message,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIDeveloperMessageParam,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
    SamplingParams,
    StopReason,
    SystemMessage,
    ToolDefinition,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolInvocationResult, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.core.datatypes import AccessRule
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    ToolCall,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict_new,
    convert_openai_chat_completion_stream,
    convert_tooldef_to_openai_tool,
)
from llama_stack.providers.utils.kvstore import KVStore
from llama_stack.providers.utils.telemetry import tracing

from .persistence import AgentPersistence
from .safety import SafetyException, ShieldRunnerMixin

TOOLS_ATTACHMENT_KEY_REGEX = re.compile(r"__tools_attachment__=(\{.*?\})")
MEMORY_QUERY_TOOL = "knowledge_search"
WEB_SEARCH_TOOL = "web_search"
RAG_TOOL_GROUP = "builtin::rag"

logger = get_logger(name=__name__, category="agents::meta_reference")


class ChatAgent(ShieldRunnerMixin):
    def __init__(
        self,
        agent_id: str,
        agent_config: AgentConfig,
        inference_api: Inference,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        vector_io_api: VectorIO,
        persistence_store: KVStore,
        created_at: str,
        policy: list[AccessRule],
        telemetry_enabled: bool = False,
    ):
        self.agent_id = agent_id
        self.agent_config = agent_config
        self.inference_api = inference_api
        self.safety_api = safety_api
        self.vector_io_api = vector_io_api
        self.storage = AgentPersistence(agent_id, persistence_store, policy)
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api
        self.created_at = created_at
        self.telemetry_enabled = telemetry_enabled

        ShieldRunnerMixin.__init__(
            self,
            safety_api,
            input_shields=agent_config.input_shields,
            output_shields=agent_config.output_shields,
        )

    def turn_to_messages(self, turn: Turn) -> list[Message]:
        messages = []

        # NOTE: if a toolcall response is in a step, we do not add it when processing the input messages
        tool_call_ids = set()
        for step in turn.steps:
            if step.step_type == StepType.tool_execution.value:
                for response in step.tool_responses:
                    tool_call_ids.add(response.call_id)

        for m in turn.input_messages:
            msg = m.model_copy()
            # We do not want to keep adding RAG context to the input messages
            # May be this should be a parameter of the agentic instance
            # that can define its behavior in a custom way
            if isinstance(msg, UserMessage):
                msg.context = None
            if isinstance(msg, ToolResponseMessage):
                if msg.call_id in tool_call_ids:
                    # NOTE: do not add ToolResponseMessage here, we'll add them in tool_execution steps
                    continue

            messages.append(msg)

        for step in turn.steps:
            if step.step_type == StepType.inference.value:
                messages.append(step.model_response)
            elif step.step_type == StepType.tool_execution.value:
                for response in step.tool_responses:
                    messages.append(
                        ToolResponseMessage(
                            call_id=response.call_id,
                            content=response.content,
                        )
                    )
            elif step.step_type == StepType.shield_call.value:
                if step.violation:
                    # CompletionMessage itself in the ShieldResponse
                    messages.append(
                        CompletionMessage(
                            content=step.violation.user_message,
                            stop_reason=StopReason.end_of_turn,
                        )
                    )
        return messages

    async def create_session(self, name: str) -> str:
        return await self.storage.create_session(name)

    async def get_messages_from_turns(self, turns: list[Turn]) -> list[Message]:
        messages = []
        if self.agent_config.instructions != "":
            messages.append(SystemMessage(content=self.agent_config.instructions))

        for turn in turns:
            messages.extend(self.turn_to_messages(turn))
        return messages

    async def create_and_execute_turn(self, request: AgentTurnCreateRequest) -> AsyncGenerator:
        turn_id = str(uuid.uuid4())
        if self.telemetry_enabled:
            span = tracing.get_current_span()
            if span is not None:
                span.set_attribute("session_id", request.session_id)
                span.set_attribute("agent_id", self.agent_id)
                span.set_attribute("request", request.model_dump_json())
                span.set_attribute("turn_id", turn_id)
                if self.agent_config.name:
                    span.set_attribute("agent_name", self.agent_config.name)

        await self._initialize_tools(request.toolgroups)
        async for chunk in self._run_turn(request, turn_id):
            yield chunk

    async def resume_turn(self, request: AgentTurnResumeRequest) -> AsyncGenerator:
        if self.telemetry_enabled:
            span = tracing.get_current_span()
            if span is not None:
                span.set_attribute("agent_id", self.agent_id)
                span.set_attribute("session_id", request.session_id)
                span.set_attribute("request", request.model_dump_json())
                span.set_attribute("turn_id", request.turn_id)
                if self.agent_config.name:
                    span.set_attribute("agent_name", self.agent_config.name)

        await self._initialize_tools()
        async for chunk in self._run_turn(request):
            yield chunk

    async def _run_turn(
        self,
        request: AgentTurnCreateRequest | AgentTurnResumeRequest,
        turn_id: str | None = None,
    ) -> AsyncGenerator:
        assert request.stream is True, "Non-streaming not supported"

        is_resume = isinstance(request, AgentTurnResumeRequest)
        session_info = await self.storage.get_session_info(request.session_id)
        if session_info is None:
            raise SessionNotFoundError(request.session_id)

        turns = await self.storage.get_session_turns(request.session_id)
        if is_resume and len(turns) == 0:
            raise ValueError("No turns found for session")

        steps = []
        messages = await self.get_messages_from_turns(turns)
        if is_resume:
            tool_response_messages = [
                ToolResponseMessage(call_id=x.call_id, content=x.content) for x in request.tool_responses
            ]
            messages.extend(tool_response_messages)
            last_turn = turns[-1]
            last_turn_messages = self.turn_to_messages(last_turn)
            last_turn_messages = [
                x for x in last_turn_messages if isinstance(x, UserMessage) or isinstance(x, ToolResponseMessage)
            ]
            last_turn_messages.extend(tool_response_messages)

            # get steps from the turn
            steps = last_turn.steps

            # mark tool execution step as complete
            # if there's no tool execution in progress step (due to storage, or tool call parsing on client),
            # we'll create a new tool execution step with current time
            in_progress_tool_call_step = await self.storage.get_in_progress_tool_call_step(
                request.session_id, request.turn_id
            )
            now = datetime.now(UTC).isoformat()
            tool_execution_step = ToolExecutionStep(
                step_id=(in_progress_tool_call_step.step_id if in_progress_tool_call_step else str(uuid.uuid4())),
                turn_id=request.turn_id,
                tool_calls=(in_progress_tool_call_step.tool_calls if in_progress_tool_call_step else []),
                tool_responses=request.tool_responses,
                completed_at=now,
                started_at=(in_progress_tool_call_step.started_at if in_progress_tool_call_step else now),
            )
            steps.append(tool_execution_step)
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.tool_execution.value,
                        step_id=tool_execution_step.step_id,
                        step_details=tool_execution_step,
                    )
                )
            )
            input_messages = last_turn.input_messages

            turn_id = request.turn_id
            start_time = last_turn.started_at
        else:
            messages.extend(request.messages)
            start_time = datetime.now(UTC).isoformat()
            input_messages = request.messages

        output_message = None
        async for chunk in self.run(
            session_id=request.session_id,
            turn_id=turn_id,
            input_messages=messages,
            sampling_params=self.agent_config.sampling_params,
            stream=request.stream,
            documents=request.documents if not is_resume else None,
        ):
            if isinstance(chunk, CompletionMessage):
                output_message = chunk
                continue

            assert isinstance(chunk, AgentTurnResponseStreamChunk), f"Unexpected type {type(chunk)}"
            event = chunk.event
            if event.payload.event_type == AgentTurnResponseEventType.step_complete.value:
                steps.append(event.payload.step_details)

            yield chunk

        assert output_message is not None

        turn = Turn(
            turn_id=turn_id,
            session_id=request.session_id,
            input_messages=input_messages,
            output_message=output_message,
            started_at=start_time,
            completed_at=datetime.now(UTC).isoformat(),
            steps=steps,
        )
        await self.storage.add_turn_to_session(request.session_id, turn)
        if output_message.tool_calls:
            chunk = AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseTurnAwaitingInputPayload(
                        turn=turn,
                    )
                )
            )
        else:
            chunk = AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseTurnCompletePayload(
                        turn=turn,
                    )
                )
            )

        yield chunk

    async def run(
        self,
        session_id: str,
        turn_id: str,
        input_messages: list[Message],
        sampling_params: SamplingParams,
        stream: bool = False,
        documents: list[Document] | None = None,
    ) -> AsyncGenerator:
        # Doing async generators makes downstream code much simpler and everything amenable to
        # streaming. However, it also makes things complicated here because AsyncGenerators cannot
        # return a "final value" for the `yield from` statement. we simulate that by yielding a
        # final boolean (to see whether an exception happened) and then explicitly testing for it.

        if len(self.input_shields) > 0:
            async for res in self.run_multiple_shields_wrapper(
                turn_id, input_messages, self.input_shields, "user-input"
            ):
                if isinstance(res, bool):
                    return
                else:
                    yield res

        async for res in self._run(
            session_id,
            turn_id,
            input_messages,
            sampling_params,
            stream,
            documents,
        ):
            if isinstance(res, bool):
                return
            elif isinstance(res, CompletionMessage):
                final_response = res
                break
            else:
                yield res

        assert final_response is not None
        # for output shields run on the full input and output combination
        messages = input_messages + [final_response]

        if len(self.output_shields) > 0:
            async for res in self.run_multiple_shields_wrapper(
                turn_id, messages, self.output_shields, "assistant-output"
            ):
                if isinstance(res, bool):
                    return
                else:
                    yield res

        yield final_response

    async def run_multiple_shields_wrapper(
        self,
        turn_id: str,
        messages: list[Message],
        shields: list[str],
        touchpoint: str,
    ) -> AsyncGenerator:
        async with tracing.span("run_shields") as span:
            if self.telemetry_enabled and span is not None:
                span.set_attribute("input", [m.model_dump_json() for m in messages])
                if len(shields) == 0:
                    span.set_attribute("output", "no shields")

            if len(shields) == 0:
                return

            step_id = str(uuid.uuid4())
            shield_call_start_time = datetime.now(UTC).isoformat()
            try:
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepStartPayload(
                            step_type=StepType.shield_call.value,
                            step_id=step_id,
                            metadata=dict(touchpoint=touchpoint),
                        )
                    )
                )
                await self.run_multiple_shields(messages, shields)

            except SafetyException as e:
                yield AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepCompletePayload(
                            step_type=StepType.shield_call.value,
                            step_id=step_id,
                            step_details=ShieldCallStep(
                                step_id=step_id,
                                turn_id=turn_id,
                                violation=e.violation,
                                started_at=shield_call_start_time,
                                completed_at=datetime.now(UTC).isoformat(),
                            ),
                        )
                    )
                )
                if self.telemetry_enabled and span is not None:
                    span.set_attribute("output", e.violation.model_dump_json())

                yield CompletionMessage(
                    content=str(e),
                    stop_reason=StopReason.end_of_turn,
                )
                yield False

            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.shield_call.value,
                        step_id=step_id,
                        step_details=ShieldCallStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            violation=None,
                            started_at=shield_call_start_time,
                            completed_at=datetime.now(UTC).isoformat(),
                        ),
                    )
                )
            )
            if self.telemetry_enabled and span is not None:
                span.set_attribute("output", "no violations")

    async def _run(
        self,
        session_id: str,
        turn_id: str,
        input_messages: list[Message],
        sampling_params: SamplingParams,
        stream: bool = False,
        documents: list[Document] | None = None,
    ) -> AsyncGenerator:
        # if document is passed in a turn, we parse the raw text of the document
        # and sent it as a user message
        if documents:
            contexts = []
            for document in documents:
                raw_document_text = await get_raw_document_text(document)
                contexts.append(raw_document_text)

            attached_context = "\n".join(contexts)
            if isinstance(input_messages[-1].content, str):
                input_messages[-1].content += attached_context
            elif isinstance(input_messages[-1].content, list):
                input_messages[-1].content.append(TextContentItem(text=attached_context))
            else:
                input_messages[-1].content = [
                    input_messages[-1].content,
                    TextContentItem(text=attached_context),
                ]

        session_info = await self.storage.get_session_info(session_id)
        # if the session has a memory bank id, let the memory tool use it
        if session_info and session_info.vector_db_id:
            for tool_name in self.tool_name_to_args.keys():
                if tool_name == MEMORY_QUERY_TOOL:
                    if "vector_db_ids" not in self.tool_name_to_args[tool_name]:
                        self.tool_name_to_args[tool_name]["vector_db_ids"] = [session_info.vector_db_id]
                    else:
                        self.tool_name_to_args[tool_name]["vector_db_ids"].append(session_info.vector_db_id)

        output_attachments = []

        n_iter = await self.storage.get_num_infer_iters_in_turn(session_id, turn_id) or 0

        # Build a map of custom tools to their definitions for faster lookup
        client_tools = {}
        for tool in self.agent_config.client_tools:
            client_tools[tool.name] = tool
        while True:
            step_id = str(uuid.uuid4())
            inference_start_time = datetime.now(UTC).isoformat()
            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepStartPayload(
                        step_type=StepType.inference.value,
                        step_id=step_id,
                    )
                )
            )

            tool_calls = []
            content = ""
            stop_reason: StopReason | None = None

            async with tracing.span("inference") as span:
                if self.telemetry_enabled and span is not None:
                    if self.agent_config.name:
                        span.set_attribute("agent_name", self.agent_config.name)

                def _serialize_nested(value):
                    """Recursively serialize nested Pydantic models to dicts."""
                    from pydantic import BaseModel

                    if isinstance(value, BaseModel):
                        return value.model_dump(mode="json")
                    elif isinstance(value, dict):
                        return {k: _serialize_nested(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        return [_serialize_nested(item) for item in value]
                    else:
                        return value

                def _add_type(openai_msg: dict) -> OpenAIMessageParam:
                    # Serialize any nested Pydantic models to plain dicts
                    openai_msg = _serialize_nested(openai_msg)

                    role = openai_msg.get("role")
                    if role == "user":
                        return OpenAIUserMessageParam(**openai_msg)
                    elif role == "system":
                        return OpenAISystemMessageParam(**openai_msg)
                    elif role == "assistant":
                        return OpenAIAssistantMessageParam(**openai_msg)
                    elif role == "tool":
                        return OpenAIToolMessageParam(**openai_msg)
                    elif role == "developer":
                        return OpenAIDeveloperMessageParam(**openai_msg)
                    else:
                        raise ValueError(f"Unknown message role: {role}")

                # Convert messages to OpenAI format
                openai_messages: list[OpenAIMessageParam] = [
                    _add_type(await convert_message_to_openai_dict_new(message)) for message in input_messages
                ]

                # Convert tool definitions to OpenAI format
                openai_tools = [convert_tooldef_to_openai_tool(x) for x in (self.tool_defs or [])]

                # Extract tool_choice from tool_config for OpenAI compatibility
                # Note: tool_choice can only be provided when tools are also provided
                tool_choice = None
                if openai_tools and self.agent_config.tool_config and self.agent_config.tool_config.tool_choice:
                    tc = self.agent_config.tool_config.tool_choice
                    tool_choice_str = tc.value if hasattr(tc, "value") else str(tc)
                    # Convert tool_choice to OpenAI format
                    if tool_choice_str in ("auto", "none", "required"):
                        tool_choice = tool_choice_str
                    else:
                        # It's a specific tool name, wrap it in the proper format
                        tool_choice = {"type": "function", "function": {"name": tool_choice_str}}

                # Convert sampling params to OpenAI format (temperature, top_p, max_tokens)
                temperature = getattr(getattr(sampling_params, "strategy", None), "temperature", None)
                top_p = getattr(getattr(sampling_params, "strategy", None), "top_p", None)
                max_tokens = getattr(sampling_params, "max_tokens", None)

                # Use OpenAI chat completion
                params = OpenAIChatCompletionRequestWithExtraBody(
                    model=self.agent_config.model,
                    messages=openai_messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice=tool_choice,
                    response_format=self.agent_config.response_format,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True,
                )
                openai_stream = await self.inference_api.openai_chat_completion(params)

                # Convert OpenAI stream back to Llama Stack format
                response_stream = convert_openai_chat_completion_stream(
                    openai_stream, enable_incremental_tool_calls=True
                )

                async for chunk in response_stream:
                    event = chunk.event
                    if event.event_type == ChatCompletionResponseEventType.start:
                        continue
                    elif event.event_type == ChatCompletionResponseEventType.complete:
                        stop_reason = event.stop_reason or StopReason.end_of_turn
                        continue

                    delta = event.delta
                    if delta.type == "tool_call":
                        if delta.parse_status == ToolCallParseStatus.succeeded:
                            tool_calls.append(delta.tool_call)
                        elif delta.parse_status == ToolCallParseStatus.failed:
                            # If we cannot parse the tools, set the content to the unparsed raw text
                            content = str(delta.tool_call)
                        if stream:
                            yield AgentTurnResponseStreamChunk(
                                event=AgentTurnResponseEvent(
                                    payload=AgentTurnResponseStepProgressPayload(
                                        step_type=StepType.inference.value,
                                        step_id=step_id,
                                        delta=delta,
                                    )
                                )
                            )

                    elif delta.type == "text":
                        content += delta.text
                        if stream and event.stop_reason is None:
                            yield AgentTurnResponseStreamChunk(
                                event=AgentTurnResponseEvent(
                                    payload=AgentTurnResponseStepProgressPayload(
                                        step_type=StepType.inference.value,
                                        step_id=step_id,
                                        delta=delta,
                                    )
                                )
                            )
                    else:
                        raise ValueError(f"Unexpected delta type {type(delta)}")

                if self.telemetry_enabled and span is not None:
                    span.set_attribute("stop_reason", stop_reason or StopReason.end_of_turn)
                    span.set_attribute(
                        "input",
                        json.dumps([json.loads(m.model_dump_json()) for m in input_messages]),
                    )
                    output_attr = json.dumps(
                        {
                            "content": content,
                            "tool_calls": [json.loads(t.model_dump_json()) for t in tool_calls],
                        }
                    )
                    span.set_attribute("output", output_attr)

            n_iter += 1
            await self.storage.set_num_infer_iters_in_turn(session_id, turn_id, n_iter)

            stop_reason = stop_reason or StopReason.out_of_tokens

            # If tool calls are parsed successfully,
            # if content is not made null the tool call str will also be in the content
            # and tokens will have tool call syntax included twice
            if tool_calls:
                content = ""

            message = CompletionMessage(
                content=content,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
            )

            yield AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.inference.value,
                        step_id=step_id,
                        step_details=InferenceStep(
                            # somewhere deep, we are re-assigning message or closing over some
                            # variable which causes message to mutate later on. fix with a
                            # `deepcopy` for now, but this is symptomatic of a deeper issue.
                            step_id=step_id,
                            turn_id=turn_id,
                            model_response=copy.deepcopy(message),
                            started_at=inference_start_time,
                            completed_at=datetime.now(UTC).isoformat(),
                        ),
                    )
                )
            )

            if n_iter >= self.agent_config.max_infer_iters:
                logger.info(f"done with MAX iterations ({n_iter}), exiting.")
                # NOTE: mark end_of_turn to indicate to client that we are done with the turn
                # Do not continue the tool call loop after this point
                message.stop_reason = StopReason.end_of_turn
                yield message
                break

            if stop_reason == StopReason.out_of_tokens:
                logger.info("out of token budget, exiting.")
                yield message
                break

            if len(message.tool_calls) == 0:
                if stop_reason == StopReason.end_of_turn:
                    # TODO: UPDATE RETURN TYPE TO SEND A TUPLE OF (MESSAGE, ATTACHMENTS)
                    if len(output_attachments) > 0:
                        if isinstance(message.content, list):
                            message.content += output_attachments
                        else:
                            message.content = [message.content] + output_attachments
                    yield message
                else:
                    logger.debug(f"completion message with EOM (iter: {n_iter}): {str(message)}")
                    input_messages = input_messages + [message]
            else:
                input_messages = input_messages + [message]

                # Process tool calls in the message
                client_tool_calls = []
                non_client_tool_calls = []

                # Separate client and non-client tool calls
                for tool_call in message.tool_calls:
                    if tool_call.tool_name in client_tools:
                        client_tool_calls.append(tool_call)
                    else:
                        non_client_tool_calls.append(tool_call)

                # Process non-client tool calls first
                for tool_call in non_client_tool_calls:
                    step_id = str(uuid.uuid4())
                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepStartPayload(
                                step_type=StepType.tool_execution.value,
                                step_id=step_id,
                            )
                        )
                    )

                    yield AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseStepProgressPayload(
                                step_type=StepType.tool_execution.value,
                                step_id=step_id,
                                delta=ToolCallDelta(
                                    parse_status=ToolCallParseStatus.in_progress,
                                    tool_call=tool_call,
                                ),
                            )
                        )
                    )

                    # Execute the tool call
                    async with tracing.span(
                        "tool_execution",
                        {
                            "tool_name": tool_call.tool_name,
                            "input": message.model_dump_json(),
                        }
                        if self.telemetry_enabled
                        else {},
                    ) as span:
                        tool_execution_start_time = datetime.now(UTC).isoformat()
                        tool_result = await self.execute_tool_call_maybe(
                            session_id,
                            tool_call,
                        )
                        if tool_result.content is None:
                            raise ValueError(
                                f"Tool call result (id: {tool_call.call_id}, name: {tool_call.tool_name}) does not have any content"
                            )
                        result_message = ToolResponseMessage(
                            call_id=tool_call.call_id,
                            content=tool_result.content,
                        )
                        if self.telemetry_enabled and span is not None:
                            span.set_attribute("output", result_message.model_dump_json())

                        # Store tool execution step
                        tool_execution_step = ToolExecutionStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            tool_calls=[tool_call],
                            tool_responses=[
                                ToolResponse(
                                    call_id=tool_call.call_id,
                                    tool_name=tool_call.tool_name,
                                    content=tool_result.content,
                                    metadata=tool_result.metadata,
                                )
                            ],
                            started_at=tool_execution_start_time,
                            completed_at=datetime.now(UTC).isoformat(),
                        )

                        # Yield the step completion event
                        yield AgentTurnResponseStreamChunk(
                            event=AgentTurnResponseEvent(
                                payload=AgentTurnResponseStepCompletePayload(
                                    step_type=StepType.tool_execution.value,
                                    step_id=step_id,
                                    step_details=tool_execution_step,
                                )
                            )
                        )

                        # Add the result message to input_messages for the next iteration
                        input_messages.append(result_message)

                        # TODO: add tool-input touchpoint and a "start" event for this step also
                        # but that needs a lot more refactoring of Tool code potentially
                        if (type(result_message.content) is str) and (
                            out_attachment := _interpret_content_as_attachment(result_message.content)
                        ):
                            # NOTE: when we push this message back to the model, the model may ignore the
                            # attached file path etc. since the model is trained to only provide a user message
                            # with the summary. We keep all generated attachments and then attach them to final message
                            output_attachments.append(out_attachment)

                # If there are client tool calls, yield a message with only those tool calls
                if client_tool_calls:
                    await self.storage.set_in_progress_tool_call_step(
                        session_id,
                        turn_id,
                        ToolExecutionStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            tool_calls=client_tool_calls,
                            tool_responses=[],
                            started_at=datetime.now(UTC).isoformat(),
                        ),
                    )

                    # Create a copy of the message with only client tool calls
                    client_message = message.model_copy(deep=True)
                    client_message.tool_calls = client_tool_calls
                    # NOTE: mark end_of_message to indicate to client that it may
                    # call the tool and continue the conversation with the tool's response.
                    client_message.stop_reason = StopReason.end_of_message

                    # Yield the message with client tool calls
                    yield client_message
                    return

    async def _initialize_tools(
        self,
        toolgroups_for_turn: list[AgentToolGroup] | None = None,
    ) -> None:
        toolgroup_to_args = {}
        for toolgroup in (self.agent_config.toolgroups or []) + (toolgroups_for_turn or []):
            if isinstance(toolgroup, AgentToolGroupWithArgs):
                tool_group_name, _ = self._parse_toolgroup_name(toolgroup.name)
                toolgroup_to_args[tool_group_name] = toolgroup.args

        # Determine which tools to include
        tool_groups_to_include = toolgroups_for_turn or self.agent_config.toolgroups or []
        agent_config_toolgroups = []
        for toolgroup in tool_groups_to_include:
            name = toolgroup.name if isinstance(toolgroup, AgentToolGroupWithArgs) else toolgroup
            if name not in agent_config_toolgroups:
                agent_config_toolgroups.append(name)

        toolgroup_to_args = toolgroup_to_args or {}

        tool_name_to_def = {}
        tool_name_to_args = {}

        for tool_def in self.agent_config.client_tools:
            if tool_name_to_def.get(tool_def.name, None):
                raise ValueError(f"Tool {tool_def.name} already exists")

            # Use input_schema from ToolDef directly
            tool_name_to_def[tool_def.name] = ToolDefinition(
                tool_name=tool_def.name,
                description=tool_def.description,
                input_schema=tool_def.input_schema,
            )
        for toolgroup_name_with_maybe_tool_name in agent_config_toolgroups:
            toolgroup_name, input_tool_name = self._parse_toolgroup_name(toolgroup_name_with_maybe_tool_name)
            tools = await self.tool_groups_api.list_tools(toolgroup_id=toolgroup_name)
            if not tools.data:
                available_tool_groups = ", ".join(
                    [t.identifier for t in (await self.tool_groups_api.list_tool_groups()).data]
                )
                raise ValueError(f"Toolgroup {toolgroup_name} not found, available toolgroups: {available_tool_groups}")
            if input_tool_name is not None and not any(tool.name == input_tool_name for tool in tools.data):
                raise ValueError(
                    f"Tool {input_tool_name} not found in toolgroup {toolgroup_name}. Available tools: {', '.join([tool.name for tool in tools.data])}"
                )

            for tool_def in tools.data:
                if toolgroup_name.startswith("builtin") and toolgroup_name != RAG_TOOL_GROUP:
                    identifier: str | BuiltinTool | None = tool_def.name
                    if identifier == "web_search":
                        identifier = BuiltinTool.brave_search
                    else:
                        identifier = BuiltinTool(identifier)
                else:
                    # add if tool_name is unspecified or the tool_def identifier is the same as the tool_name
                    if input_tool_name in (None, tool_def.name):
                        identifier = tool_def.name
                    else:
                        identifier = None

                if tool_name_to_def.get(identifier, None):
                    raise ValueError(f"Tool {identifier} already exists")
                if identifier:
                    tool_name_to_def[identifier] = ToolDefinition(
                        tool_name=identifier,
                        description=tool_def.description,
                        input_schema=tool_def.input_schema,
                    )
                    tool_name_to_args[identifier] = toolgroup_to_args.get(toolgroup_name, {})

        self.tool_defs, self.tool_name_to_args = (
            list(tool_name_to_def.values()),
            tool_name_to_args,
        )

    def _parse_toolgroup_name(self, toolgroup_name_with_maybe_tool_name: str) -> tuple[str, str | None]:
        """Parse a toolgroup name into its components.

        Args:
            toolgroup_name: The toolgroup name to parse (e.g. "builtin::rag/knowledge_search")

        Returns:
            A tuple of (tool_type, tool_group, tool_name)
        """
        split_names = toolgroup_name_with_maybe_tool_name.split("/")
        if len(split_names) == 2:
            # e.g. "builtin::rag"
            tool_group, tool_name = split_names
        else:
            tool_group, tool_name = split_names[0], None
        return tool_group, tool_name

    async def execute_tool_call_maybe(
        self,
        session_id: str,
        tool_call: ToolCall,
    ) -> ToolInvocationResult:
        tool_name = tool_call.tool_name
        registered_tool_names = [tool_def.tool_name for tool_def in self.tool_defs]
        if tool_name not in registered_tool_names:
            raise ValueError(
                f"Tool {tool_name} not found in provided tools, registered tools: {', '.join([str(x) for x in registered_tool_names])}"
            )
        if isinstance(tool_name, BuiltinTool):
            if tool_name == BuiltinTool.brave_search:
                tool_name_str = WEB_SEARCH_TOOL
            else:
                tool_name_str = tool_name.value
        else:
            tool_name_str = tool_name

        logger.info(f"executing tool call: {tool_name_str} with args: {tool_call.arguments}")

        try:
            args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse arguments for tool call: {tool_call.arguments}") from e

        result = await self.tool_runtime_api.invoke_tool(
            tool_name=tool_name_str,
            kwargs={
                "session_id": session_id,
                # get the arguments generated by the model and augment with toolgroup arg overrides for the agent
                **args,
                **self.tool_name_to_args.get(tool_name_str, {}),
            },
        )
        logger.debug(f"tool call {tool_name_str} completed with result: {result}")
        return result


async def load_data_from_url(url: str) -> str:
    if url.startswith("http"):
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            resp = r.text
            return resp
    raise ValueError(f"Unexpected URL: {type(url)}")


async def get_raw_document_text(document: Document) -> str:
    # Handle deprecated text/yaml mime type with warning
    if document.mime_type == "text/yaml":
        warnings.warn(
            "The 'text/yaml' MIME type is deprecated. Please use 'application/yaml' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    elif not (document.mime_type.startswith("text/") or document.mime_type in ("application/yaml", "application/json")):
        raise ValueError(f"Unexpected document mime type: {document.mime_type}")

    if isinstance(document.content, URL):
        return await load_data_from_url(document.content.uri)
    elif isinstance(document.content, str):
        return document.content
    elif isinstance(document.content, TextContentItem):
        return document.content.text
    else:
        raise ValueError(f"Unexpected document content type: {type(document.content)}")


def _interpret_content_as_attachment(
    content: str,
) -> Attachment | None:
    match = re.search(TOOLS_ATTACHMENT_KEY_REGEX, content)
    if match:
        snippet = match.group(1)
        data = json.loads(snippet)
        return Attachment(
            url=URL(uri="file://" + data["filepath"]),
            mime_type=data["mimetype"],
        )

    return None
