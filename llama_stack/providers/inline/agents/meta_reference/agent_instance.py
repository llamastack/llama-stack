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
from typing import Any

import httpx

from llama_stack.apis.agents import (
    AgentConfig,
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
    OpenAIResponseInputTool,
    ShieldCallStep,
    Step,
    StepType,
    ToolExecutionStep,
    Turn,
)
from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolWebSearch,
)
from llama_stack.apis.common.content_types import URL, ToolCallDelta, ToolCallParseStatus
from llama_stack.apis.common.errors import SessionNotFoundError
from llama_stack.apis.inference import (
    ChatCompletionResponseEventType,
    CompletionMessage,
    Inference,
    Message,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionMessageContent,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIDeveloperMessageParam,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
    StopReason,
    SystemMessage,
    ToolDefinition,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolDef, ToolGroups, ToolInvocationResult, ToolRuntime
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


def _map_finish_reason_to_stop_reason(finish_reason: str | None) -> StopReason:
    if finish_reason == "length":
        return StopReason.out_of_tokens
    if finish_reason == "tool_calls":
        return StopReason.end_of_message
    # Default to end_of_turn for unknown or "stop"
    return StopReason.end_of_turn


def _map_stop_reason_to_finish_reason(stop_reason: StopReason | None) -> str | None:
    if stop_reason == StopReason.out_of_tokens:
        return "length"
    if stop_reason == StopReason.end_of_message:
        return "tool_calls"
    if stop_reason == StopReason.end_of_turn:
        return "stop"
    return None


def _openai_tool_call_to_legacy(tool_call: OpenAIChatCompletionToolCall) -> ToolCall:
    name = None
    if tool_call.function and tool_call.function.name:
        name = tool_call.function.name
    return ToolCall(
        call_id=tool_call.id or f"call_{uuid.uuid4()}",
        tool_name=name or "",
        arguments=tool_call.function.arguments if tool_call.function and tool_call.function.arguments else "{}",
    )


def _legacy_tool_call_to_openai(tool_call: ToolCall, index: int | None = None) -> OpenAIChatCompletionToolCall:
    function_name = (
        tool_call.tool_name if not isinstance(tool_call.tool_name, BuiltinTool) else tool_call.tool_name.value
    )
    return OpenAIChatCompletionToolCall(
        index=index,
        id=tool_call.call_id,
        function=OpenAIChatCompletionToolCallFunction(
            name=function_name,
            arguments=tool_call.arguments,
        ),
    )


def _tool_response_message_to_openai(response: ToolResponseMessage) -> OpenAIToolMessageParam:
    content = _coerce_to_text(response.content)
    return OpenAIToolMessageParam(
        tool_call_id=response.call_id,
        content=content,
    )


def _openai_message_content_to_text(
    content: OpenAIChatCompletionMessageContent,
) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, OpenAIChatCompletionContentPartTextParam):
            parts.append(item.text)
        elif isinstance(item, OpenAIChatCompletionContentPartImageParam) and item.image_url:
            if item.image_url.url:
                parts.append(item.image_url.url)
    return "\n".join(parts)


def _append_text_to_openai_message(message: OpenAIMessageParam, text: str) -> None:
    if not text:
        return
    if isinstance(message, OpenAIUserMessageParam):
        content = message.content
        if content is None or content == "":
            message.content = text
        elif isinstance(content, str):
            message.content = f"{content}\n{text}"
        else:
            content.append(OpenAIChatCompletionContentPartTextParam(text=text))


def _coerce_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_coerce_to_text(item) for item in content)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "image"):
        image = content.image
        if hasattr(image, "url") and image.url:
            return getattr(image.url, "uri", "")
    return str(content)


def _openai_message_param_to_legacy(message: OpenAIMessageParam) -> Message:
    if isinstance(message, OpenAIUserMessageParam):
        return UserMessage(content=_openai_message_content_to_text(message.content))
    if isinstance(message, OpenAISystemMessageParam):
        return SystemMessage(content=_openai_message_content_to_text(message.content))
    if isinstance(message, OpenAIToolMessageParam):
        return ToolResponseMessage(
            call_id=message.tool_call_id,
            content=_openai_message_content_to_text(message.content),
        )
    if isinstance(message, OpenAIDeveloperMessageParam):
        # Map developer messages to user role for legacy compatibility
        return UserMessage(content=_openai_message_content_to_text(message.content))
    if isinstance(message, OpenAIAssistantMessageParam):
        tool_calls = [_openai_tool_call_to_legacy(tool_call) for tool_call in message.tool_calls or []]
        return CompletionMessage(
            content=_openai_message_content_to_text(message.content) if message.content is not None else "",
            stop_reason=StopReason.end_of_turn,
            tool_calls=tool_calls,
        )
    raise ValueError(f"Unsupported OpenAI message type: {type(message)}")


async def _legacy_message_to_openai(message: Message) -> OpenAIMessageParam:
    openai_dict = await convert_message_to_openai_dict_new(message)
    role = openai_dict.get("role")
    if role == "user":
        return OpenAIUserMessageParam(**openai_dict)
    if role == "system":
        return OpenAISystemMessageParam(**openai_dict)
    if role == "assistant":
        return OpenAIAssistantMessageParam(**openai_dict)
    if role == "tool":
        return OpenAIToolMessageParam(**openai_dict)
    if role == "developer":
        return OpenAIDeveloperMessageParam(**openai_dict)
    raise ValueError(f"Unsupported OpenAI message role: {role}")


async def _completion_to_openai_assistant(
    completion: CompletionMessage,
) -> tuple[OpenAIAssistantMessageParam, str | None]:
    assistant_param = await _legacy_message_to_openai(completion)
    assert isinstance(assistant_param, OpenAIAssistantMessageParam)
    finish_reason = _map_stop_reason_to_finish_reason(completion.stop_reason)
    return assistant_param, finish_reason


def _client_tool_to_tool_definition(tool: OpenAIResponseInputTool | ToolDef) -> ToolDefinition:
    if isinstance(tool, ToolDef):
        return ToolDefinition(
            tool_name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )
    if getattr(tool, "type", None) == "function":
        return ToolDefinition(
            tool_name=tool.name,
            description=getattr(tool, "description", None),
            input_schema=getattr(tool, "parameters", None),
        )
    raise ValueError(f"Unsupported client tool type '{getattr(tool, 'type', None)}' for agent configuration")


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

        self.tool_defs: list[ToolDefinition] = []
        self.tool_name_to_args: dict[str | BuiltinTool, dict[str, Any]] = {}
        self.client_tools_config: list[OpenAIResponseInputTool | ToolDef] = []

        ShieldRunnerMixin.__init__(
            self,
            safety_api,
            input_shields=agent_config.input_shields,
            output_shields=agent_config.output_shields,
        )

    def _resolve_generation_options(
        self,
        request: AgentTurnCreateRequest | AgentTurnResumeRequest,
    ) -> dict[str, Any]:
        def _pick(attr: str) -> Any:
            value = getattr(request, attr, None)
            if value is not None:
                return value
            return getattr(self.agent_config, attr)

        temperature = _pick("temperature")
        top_p = _pick("top_p")
        max_output_tokens = _pick("max_output_tokens")
        stop = _pick("stop")

        # Ensure we don't share mutable defaults
        if isinstance(stop, list):
            stop = list(stop)

        return {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "stop": stop,
        }

    def turn_to_messages(self, turn: Turn) -> list[OpenAIMessageParam]:
        messages: list[OpenAIMessageParam] = []

        tool_response_ids = {
            response.tool_call_id
            for step in turn.steps
            if step.step_type == StepType.tool_execution.value
            for response in step.tool_responses
        }

        for message in turn.input_messages:
            copied = message.model_copy(deep=True)
            if isinstance(copied, OpenAIToolMessageParam) and copied.tool_call_id in tool_response_ids:
                # Skip tool responses; they will be reintroduced from the execution step
                continue
            messages.append(copied)

        for step in turn.steps:
            if step.step_type == StepType.inference.value:
                messages.append(step.model_response.model_copy(deep=True))
            elif step.step_type == StepType.tool_execution.value:
                for response in step.tool_responses:
                    messages.append(response.model_copy(deep=True))
            elif step.step_type == StepType.shield_call.value and step.violation:
                assistant_msg = OpenAIAssistantMessageParam(
                    content=str(step.violation.user_message),
                )
                messages.append(assistant_msg)

        return messages

    async def create_session(self, name: str) -> str:
        return await self.storage.create_session(name)

    async def get_messages_from_turns(self, turns: list[Turn]) -> list[OpenAIMessageParam]:
        messages: list[OpenAIMessageParam] = []
        if self.agent_config.instructions:
            messages.append(OpenAISystemMessageParam(content=self.agent_config.instructions))

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

        await self._initialize_tools(request.tools)
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

        steps: list[Step] = []
        history_openai = await self.get_messages_from_turns(turns)

        if turn_id is None:
            turn_id = request.turn_id

        if is_resume:
            tool_response_messages = [resp.model_copy(deep=True) for resp in request.tool_responses]
            history_openai.extend(tool_response_messages)

            last_turn = turns[-1]
            steps = list(last_turn.steps)

            in_progress_tool_call_step = await self.storage.get_in_progress_tool_call_step(
                request.session_id, request.turn_id
            )
            now = datetime.now(UTC).isoformat()
            tool_execution_step = ToolExecutionStep(
                step_id=(in_progress_tool_call_step.step_id if in_progress_tool_call_step else str(uuid.uuid4())),
                turn_id=request.turn_id,
                tool_calls=(in_progress_tool_call_step.tool_calls if in_progress_tool_call_step else []),
                tool_responses=tool_response_messages,
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

            input_messages_openai = [msg.model_copy(deep=True) for msg in last_turn.input_messages]
            start_time = last_turn.started_at
        else:
            new_messages = [msg.model_copy(deep=True) for msg in request.messages]
            history_openai.extend(new_messages)
            input_messages_openai = new_messages
            start_time = datetime.now(UTC).isoformat()

        generation_options = self._resolve_generation_options(request)

        output_completion: CompletionMessage | None = None
        output_finish_reason: str | None = None
        output_assistant_message: OpenAIAssistantMessageParam | None = None
        async for chunk in self.run(
            session_id=request.session_id,
            turn_id=turn_id,
            input_messages=history_openai,
            stream=request.stream,
            documents=request.documents if not is_resume else None,
            temperature=generation_options["temperature"],
            top_p=generation_options["top_p"],
            max_output_tokens=generation_options["max_output_tokens"],
            stop=generation_options["stop"],
        ):
            if isinstance(chunk, CompletionMessage):
                output_completion = chunk
                output_assistant_message, output_finish_reason = await _completion_to_openai_assistant(chunk)
                continue

            assert isinstance(chunk, AgentTurnResponseStreamChunk), f"Unexpected type {type(chunk)}"
            event = chunk.event
            if event.payload.event_type == AgentTurnResponseEventType.step_complete.value:
                steps.append(event.payload.step_details)

            yield chunk

        assert output_completion is not None
        assert output_assistant_message is not None

        turn = Turn(
            turn_id=turn_id,
            session_id=request.session_id,
            input_messages=input_messages_openai,
            output_message=output_assistant_message,
            finish_reason=output_finish_reason,
            started_at=start_time,
            completed_at=datetime.now(UTC).isoformat(),
            steps=steps,
        )
        await self.storage.add_turn_to_session(request.session_id, turn)
        if output_assistant_message.tool_calls:
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
        input_messages: list[OpenAIMessageParam],
        stream: bool = False,
        documents: list[Document] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: list[str] | None = None,
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
            stream,
            documents,
            temperature,
            top_p,
            max_output_tokens,
            stop,
        ):
            if isinstance(res, bool):
                return
            elif isinstance(res, CompletionMessage):
                final_response = res
                break
            else:
                yield res

        assert final_response is not None
        final_assistant, final_finish_reason = await _completion_to_openai_assistant(copy.deepcopy(final_response))
        # for output shields run on the full input and output combination
        messages = input_messages + [final_assistant.model_copy(deep=True)]

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
        messages: list[OpenAIMessageParam],
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
                legacy_messages = [_openai_message_param_to_legacy(m) for m in messages]
                await self.run_multiple_shields(legacy_messages, shields)

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
        input_messages: list[OpenAIMessageParam],
        stream: bool = False,
        documents: list[Document] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncGenerator:
        conversation = [msg.model_copy(deep=True) for msg in input_messages]

        # if document is passed in a turn, hydrate the last user message with the context
        if documents and conversation:
            appended_texts = []
            for document in documents:
                raw_document_text = await get_raw_document_text(document)
                if raw_document_text:
                    appended_texts.append(raw_document_text)
            if appended_texts:
                _append_text_to_openai_message(conversation[-1], "\n".join(appended_texts))

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
        client_tools: dict[str, OpenAIResponseInputTool | ToolDef] = {}
        for tool in self.client_tools_config or []:
            if isinstance(tool, ToolDef) and tool.name:
                client_tools[tool.name] = tool
            elif getattr(tool, "type", None) == "function" and getattr(tool, "name", None):
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
                if self.telemetry_enabled and span is not None and self.agent_config.name:
                    span.set_attribute("agent_name", self.agent_config.name)

                openai_tools = [convert_tooldef_to_openai_tool(x) for x in (self.tool_defs or [])]

                tool_choice = None
                if openai_tools and self.agent_config.tool_config and self.agent_config.tool_config.tool_choice:
                    tc = self.agent_config.tool_config.tool_choice
                    tool_choice_str = tc.value if hasattr(tc, "value") else str(tc)
                    if tool_choice_str in ("auto", "none", "required"):
                        tool_choice = tool_choice_str
                    else:
                        tool_choice = {"type": "function", "function": {"name": tool_choice_str}}

                openai_stream = await self.inference_api.openai_chat_completion(
                    model=self.agent_config.model,
                    messages=[msg.model_copy(deep=True) for msg in conversation],
                    tools=openai_tools if openai_tools else None,
                    tool_choice=tool_choice,
                    response_format=self.agent_config.response_format,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_output_tokens,
                    stop=stop,
                    stream=True,
                )

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
                        json.dumps([json.loads(m.model_copy(deep=True).model_dump_json()) for m in conversation]),
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

            assistant_param, finish_reason = await _completion_to_openai_assistant(copy.deepcopy(message))

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
                            model_response=assistant_param,
                            finish_reason=finish_reason,
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

            assistant_param = assistant_param.model_copy(deep=True)

            if len(message.tool_calls) == 0:
                if stop_reason == StopReason.end_of_turn:
                    if len(output_attachments) > 0:
                        if isinstance(message.content, list):
                            message.content += output_attachments
                        else:
                            message.content = [message.content] + output_attachments
                    yield message
                else:
                    logger.debug(f"completion message with EOM (iter: {n_iter}): {str(message)}")
                    conversation.append(assistant_param)
            else:
                conversation.append(assistant_param)

                # Process tool calls in the message
                client_tool_calls = []
                non_client_tool_calls = []
                client_tool_calls_openai = []

                # Separate client and non-client tool calls
                for tool_call in message.tool_calls:
                    if tool_call.tool_name in client_tools:
                        client_tool_calls.append(tool_call)
                        client_tool_calls_openai.append(_legacy_tool_call_to_openai(tool_call))
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
                        openai_tool_call = _legacy_tool_call_to_openai(tool_call)
                        openai_tool_response = _tool_response_message_to_openai(result_message)

                        tool_execution_step = ToolExecutionStep(
                            step_id=step_id,
                            turn_id=turn_id,
                            tool_calls=[openai_tool_call],
                            tool_responses=[openai_tool_response],
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

                        # Add the result message to conversation for the next iteration
                        conversation.append(openai_tool_response)

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
                            tool_calls=client_tool_calls_openai,
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
        tools_for_turn: list[OpenAIResponseInputTool] | None = None,
    ) -> None:
        tool_name_to_def: dict[str | BuiltinTool, ToolDefinition] = {}
        tool_name_to_args: dict[str | BuiltinTool, dict[str, Any]] = {}
        client_tools_map: dict[str, OpenAIResponseInputTool | ToolDef] = {}

        def add_tool_definition(identifier: str | BuiltinTool, tool_definition: ToolDefinition) -> None:
            if identifier in tool_name_to_def:
                raise ValueError(f"Tool {identifier} already exists")
            tool_name_to_def[identifier] = tool_definition

        def add_client_tool(tool: OpenAIResponseInputTool | ToolDef) -> None:
            name = getattr(tool, "name", None)
            if isinstance(tool, ToolDef):
                name = tool.name
            if not name:
                raise ValueError("Client tools must have a name")
            if name not in client_tools_map:
                client_tools_map[name] = tool
                tool_definition = _client_tool_to_tool_definition(tool)
                add_tool_definition(tool_definition.tool_name, tool_definition)

        if self.agent_config.client_tools:
            for tool in self.agent_config.client_tools:
                add_client_tool(tool)

        effective_tools = tools_for_turn
        if effective_tools is None:
            effective_tools = self.agent_config.tools

        for tool in effective_tools or []:
            if isinstance(tool, OpenAIResponseInputToolFunction):
                add_client_tool(tool)
                continue

            resolved_tools = await self._resolve_response_tool(tool)
            for identifier, definition, args in resolved_tools:
                add_tool_definition(identifier, definition)
                if args:
                    existing_args = tool_name_to_args.get(identifier, {})
                    tool_name_to_args[identifier] = {**existing_args, **args}

        self.tool_defs = list(tool_name_to_def.values())
        self.tool_name_to_args = tool_name_to_args
        self.client_tools_config = list(client_tools_map.values())

    async def _resolve_response_tool(
        self,
        tool: OpenAIResponseInputTool,
    ) -> list[tuple[str | BuiltinTool, ToolDefinition, dict[str, Any]]]:
        if isinstance(tool, OpenAIResponseInputToolWebSearch):
            tool_def = await self.tool_groups_api.get_tool(WEB_SEARCH_TOOL)
            if tool_def is None:
                raise ValueError("web_search tool is not registered")
            identifier: str | BuiltinTool = BuiltinTool.brave_search
            return [
                (
                    identifier,
                    ToolDefinition(
                        tool_name=identifier,
                        description=tool_def.description,
                        input_schema=tool_def.input_schema,
                    ),
                    {},
                )
            ]

        if isinstance(tool, OpenAIResponseInputToolFileSearch):
            tool_def = await self.tool_groups_api.get_tool(MEMORY_QUERY_TOOL)
            if tool_def is None:
                raise ValueError("knowledge_search tool is not registered")
            args: dict[str, Any] = {
                "vector_db_ids": tool.vector_store_ids,
            }
            if tool.filters is not None:
                args["filters"] = tool.filters
            if tool.max_num_results is not None:
                args["max_num_results"] = tool.max_num_results
            if tool.ranking_options is not None:
                args["ranking_options"] = tool.ranking_options.model_dump()

            return [
                (
                    tool_def.name,
                    ToolDefinition(
                        tool_name=tool_def.name,
                        description=tool_def.description,
                        input_schema=tool_def.input_schema,
                    ),
                    args,
                )
            ]

        if isinstance(tool, OpenAIResponseInputToolMCP):
            toolgroup_id = tool.server_label
            if not toolgroup_id.startswith("mcp::"):
                toolgroup_id = f"mcp::{toolgroup_id}"
            tools = await self.tool_groups_api.list_tools(toolgroup_id=toolgroup_id)
            if not tools.data:
                raise ValueError(
                    f"No tools registered for MCP server '{tool.server_label}'. Ensure the toolgroup '{toolgroup_id}' is registered."
                )
            resolved: list[tuple[str | BuiltinTool, ToolDefinition, dict[str, Any]]] = []
            for tool_def in tools.data:
                resolved.append(
                    (
                        tool_def.name,
                        ToolDefinition(
                            tool_name=tool_def.name,
                            description=tool_def.description,
                            input_schema=tool_def.input_schema,
                        ),
                        {},
                    )
                )
            return resolved

        raise ValueError(f"Unsupported tool type '{getattr(tool, 'type', None)}' in agent configuration")

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
    return _openai_message_content_to_text(document.content)


def _interpret_content_as_attachment(
    content: str,
) -> Attachment | None:
    match = re.search(TOOLS_ATTACHMENT_KEY_REGEX, content)
    if match:
        snippet = match.group(1)
        data = json.loads(snippet)
        return Attachment(
            content=URL(uri="file://" + data["filepath"]),
            mime_type=data["mimetype"],
        )

    return None
