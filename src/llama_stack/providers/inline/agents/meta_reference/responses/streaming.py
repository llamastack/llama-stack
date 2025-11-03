# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from collections.abc import AsyncIterator
from typing import Any

from llama_stack.apis.agents.openai_responses import (
    AllowedToolsFilter,
    ApprovalFilter,
    MCPListToolsTool,
    OpenAIResponseContentPartOutputText,
    OpenAIResponseContentPartReasoningText,
    OpenAIResponseContentPartRefusal,
    OpenAIResponseError,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolMCP,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseContentPartAdded,
    OpenAIResponseObjectStreamResponseContentPartDone,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseFailed,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone,
    OpenAIResponseObjectStreamResponseIncomplete,
    OpenAIResponseObjectStreamResponseInProgress,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDone,
    OpenAIResponseObjectStreamResponseMcpListToolsCompleted,
    OpenAIResponseObjectStreamResponseMcpListToolsInProgress,
    OpenAIResponseObjectStreamResponseOutputItemAdded,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseObjectStreamResponseReasoningTextDelta,
    OpenAIResponseObjectStreamResponseReasoningTextDone,
    OpenAIResponseObjectStreamResponseRefusalDelta,
    OpenAIResponseObjectStreamResponseRefusalDone,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponsePrompt,
    OpenAIResponseText,
    OpenAIResponseUsage,
    OpenAIResponseUsageInputTokensDetails,
    OpenAIResponseUsageOutputTokensDetails,
    WebSearchToolTypes,
)
from llama_stack.apis.inference import (
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionToolCall,
    OpenAIChoice,
    OpenAIMessageParam,
)
from llama_stack.core.telemetry import tracing
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str

from .types import ChatCompletionContext, ChatCompletionResult
from .utils import (
    convert_chat_choice_to_response_message,
    is_function_tool_call,
    run_guardrails,
)

logger = get_logger(name=__name__, category="agents::meta_reference")


def convert_tooldef_to_chat_tool(tool_def):
    """Convert a ToolDef to OpenAI ChatCompletionToolParam format.

    Args:
        tool_def: ToolDef from the tools API

    Returns:
        ChatCompletionToolParam suitable for OpenAI chat completion
    """

    from llama_stack.models.llama.datatypes import ToolDefinition
    from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool

    internal_tool_def = ToolDefinition(
        tool_name=tool_def.name,
        description=tool_def.description,
        input_schema=tool_def.input_schema,
    )
    return convert_tooldef_to_openai_tool(internal_tool_def)


class StreamingResponseOrchestrator:
    def __init__(
        self,
        inference_api: Inference,
        ctx: ChatCompletionContext,
        response_id: str,
        created_at: int,
        text: OpenAIResponseText,
        max_infer_iters: int,
        tool_executor,  # Will be the tool execution logic from the main class
        instructions: str | None,
        safety_api,
        guardrail_ids: list[str] | None = None,
        prompt: OpenAIResponsePrompt | None = None,
        max_tool_calls: int | None = None,
    ):
        self.inference_api = inference_api
        self.ctx = ctx
        self.response_id = response_id
        self.created_at = created_at
        self.text = text
        self.max_infer_iters = max_infer_iters
        self.tool_executor = tool_executor
        self.safety_api = safety_api
        self.guardrail_ids = guardrail_ids or []
        self.prompt = prompt
        # System message that is inserted into the model's context
        self.instructions = instructions
        # Max number of total calls to built-in tools that can be processed in a response
        self.max_tool_calls = max_tool_calls
        self.sequence_number = 0
        # Store MCP tool mapping that gets built during tool processing
        self.mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] = (
            ctx.tool_context.previous_tools if ctx.tool_context else {}
        )
        # Track final messages after all tool executions
        self.final_messages: list[OpenAIMessageParam] = []
        # mapping for annotations
        self.citation_files: dict[str, str] = {}
        # Track accumulated usage across all inference calls
        self.accumulated_usage: OpenAIResponseUsage | None = None
        # Track if we've sent a refusal response
        self.violation_detected = False
        # Track total calls made to built-in tools
        self.accumulated_builtin_tool_calls = 0

    async def _create_refusal_response(self, violation_message: str) -> OpenAIResponseObjectStream:
        """Create a refusal response to replace streaming content."""
        refusal_content = OpenAIResponseContentPartRefusal(refusal=violation_message)

        # Create a completed refusal response
        refusal_response = OpenAIResponseObject(
            id=self.response_id,
            created_at=self.created_at,
            model=self.ctx.model,
            status="completed",
            output=[OpenAIResponseMessage(role="assistant", content=[refusal_content], type="message")],
        )

        return OpenAIResponseObjectStreamResponseCompleted(response=refusal_response)

    def _clone_outputs(self, outputs: list[OpenAIResponseOutput]) -> list[OpenAIResponseOutput]:
        cloned: list[OpenAIResponseOutput] = []
        for item in outputs:
            if hasattr(item, "model_copy"):
                cloned.append(item.model_copy(deep=True))
            else:
                cloned.append(item)
        return cloned

    def _snapshot_response(
        self,
        status: str,
        outputs: list[OpenAIResponseOutput],
        *,
        error: OpenAIResponseError | None = None,
    ) -> OpenAIResponseObject:
        return OpenAIResponseObject(
            created_at=self.created_at,
            id=self.response_id,
            model=self.ctx.model,
            object="response",
            status=status,
            output=self._clone_outputs(outputs),
            text=self.text,
            tools=self.ctx.available_tools(),
            error=error,
            usage=self.accumulated_usage,
            instructions=self.instructions,
            prompt=self.prompt,
            max_tool_calls=self.max_tool_calls,
        )

    async def create_response(self) -> AsyncIterator[OpenAIResponseObjectStream]:
        output_messages: list[OpenAIResponseOutput] = []

        # Emit response.created followed by response.in_progress to align with OpenAI streaming
        yield OpenAIResponseObjectStreamResponseCreated(
            response=self._snapshot_response("in_progress", output_messages)
        )

        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseInProgress(
            response=self._snapshot_response("in_progress", output_messages),
            sequence_number=self.sequence_number,
        )

        # Input safety validation - check messages before processing
        if self.guardrail_ids:
            combined_text = interleaved_content_as_str([msg.content for msg in self.ctx.messages])
            input_violation_message = await run_guardrails(self.safety_api, combined_text, self.guardrail_ids)
            if input_violation_message:
                logger.info(f"Input guardrail violation: {input_violation_message}")
                yield await self._create_refusal_response(input_violation_message)
                return

        async for stream_event in self._process_tools(output_messages):
            yield stream_event

        n_iter = 0
        messages = self.ctx.messages.copy()
        final_status = "completed"
        last_completion_result: ChatCompletionResult | None = None

        try:
            while True:
                # Text is the default response format for chat completion so don't need to pass it
                # (some providers don't support non-empty response_format when tools are present)
                response_format = (
                    None if getattr(self.ctx.response_format, "type", None) == "text" else self.ctx.response_format
                )
                logger.debug(f"calling openai_chat_completion with tools: {self.ctx.chat_tools}")

                params = OpenAIChatCompletionRequestWithExtraBody(
                    model=self.ctx.model,
                    messages=messages,
                    # Pydantic models are dict-compatible but mypy treats them as distinct types
                    tools=self.ctx.chat_tools,  # type: ignore[arg-type]
                    stream=True,
                    temperature=self.ctx.temperature,
                    response_format=response_format,
                    stream_options={
                        "include_usage": True,
                    },
                )
                completion_result = await self.inference_api.openai_chat_completion(params)

                # Process streaming chunks and build complete response
                completion_result_data = None
                async for stream_event_or_result in self._process_streaming_chunks(completion_result, output_messages):
                    if isinstance(stream_event_or_result, ChatCompletionResult):
                        completion_result_data = stream_event_or_result
                    else:
                        yield stream_event_or_result

                # If violation detected, skip the rest of processing since we already sent refusal
                if self.violation_detected:
                    return

                if not completion_result_data:
                    raise ValueError("Streaming chunk processor failed to return completion data")
                last_completion_result = completion_result_data
                current_response = self._build_chat_completion(completion_result_data)

                (
                    function_tool_calls,
                    non_function_tool_calls,
                    approvals,
                    next_turn_messages,
                ) = self._separate_tool_calls(current_response, messages)

                # add any approval requests required
                for tool_call in approvals:
                    async for evt in self._add_mcp_approval_request(
                        tool_call.function.name, tool_call.function.arguments, output_messages
                    ):
                        yield evt

                # Handle choices with no tool calls
                for choice in current_response.choices:
                    has_tool_calls = (
                        isinstance(choice.message, OpenAIAssistantMessageParam)
                        and choice.message.tool_calls
                        and self.ctx.response_tools
                    )
                    if not has_tool_calls:
                        output_messages.append(
                            await convert_chat_choice_to_response_message(
                                choice,
                                self.citation_files,
                                message_id=completion_result_data.message_item_id,
                            )
                        )

                # Execute tool calls and coordinate results
                async for stream_event in self._coordinate_tool_execution(
                    function_tool_calls,
                    non_function_tool_calls,
                    completion_result_data,
                    output_messages,
                    next_turn_messages,
                ):
                    yield stream_event

                messages = next_turn_messages

                if not function_tool_calls and not non_function_tool_calls:
                    break

                if function_tool_calls:
                    logger.info("Exiting inference loop since there is a function (client-side) tool call")
                    break

                n_iter += 1
                if n_iter >= self.max_infer_iters:
                    logger.info(
                        f"Exiting inference loop since iteration count({n_iter}) exceeds {self.max_infer_iters=}"
                    )
                    final_status = "incomplete"
                    break

            if last_completion_result and last_completion_result.finish_reason == "length":
                final_status = "incomplete"

        except Exception as exc:  # noqa: BLE001
            self.final_messages = messages.copy()
            self.sequence_number += 1
            error = OpenAIResponseError(code="internal_error", message=str(exc))
            failure_response = self._snapshot_response("failed", output_messages, error=error)
            yield OpenAIResponseObjectStreamResponseFailed(
                response=failure_response,
                sequence_number=self.sequence_number,
            )
            return

        self.final_messages = messages.copy()

        if final_status == "incomplete":
            self.sequence_number += 1
            final_response = self._snapshot_response("incomplete", output_messages)
            yield OpenAIResponseObjectStreamResponseIncomplete(
                response=final_response,
                sequence_number=self.sequence_number,
            )
        else:
            final_response = self._snapshot_response("completed", output_messages)
            yield OpenAIResponseObjectStreamResponseCompleted(response=final_response)

    def _separate_tool_calls(self, current_response, messages) -> tuple[list, list, list, list]:
        """Separate tool calls into function and non-function categories."""
        function_tool_calls = []
        non_function_tool_calls = []
        approvals = []
        next_turn_messages = messages.copy()

        for choice in current_response.choices:
            next_turn_messages.append(choice.message)
            logger.debug(f"Choice message content: {choice.message.content}")
            logger.debug(f"Choice message tool_calls: {choice.message.tool_calls}")

            if choice.message.tool_calls and self.ctx.response_tools:
                for tool_call in choice.message.tool_calls:
                    if is_function_tool_call(tool_call, self.ctx.response_tools):
                        function_tool_calls.append(tool_call)
                    else:
                        if self._approval_required(tool_call.function.name):
                            approval_response = self.ctx.approval_response(
                                tool_call.function.name, tool_call.function.arguments
                            )
                            if approval_response:
                                if approval_response.approve:
                                    logger.info(f"Approval granted for {tool_call.id} on {tool_call.function.name}")
                                    non_function_tool_calls.append(tool_call)
                                else:
                                    logger.info(f"Approval denied for {tool_call.id} on {tool_call.function.name}")
                                    next_turn_messages.pop()
                            else:
                                logger.info(f"Requesting approval for {tool_call.id} on {tool_call.function.name}")
                                approvals.append(tool_call)
                                next_turn_messages.pop()
                        else:
                            non_function_tool_calls.append(tool_call)

        return function_tool_calls, non_function_tool_calls, approvals, next_turn_messages

    def _accumulate_chunk_usage(self, chunk: OpenAIChatCompletionChunk) -> None:
        """Accumulate usage from a streaming chunk into the response usage format."""
        if not chunk.usage:
            return

        if self.accumulated_usage is None:
            # Convert from chat completion format to response format
            self.accumulated_usage = OpenAIResponseUsage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
                input_tokens_details=(
                    OpenAIResponseUsageInputTokensDetails(cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens)
                    if chunk.usage.prompt_tokens_details
                    else None
                ),
                output_tokens_details=(
                    OpenAIResponseUsageOutputTokensDetails(
                        reasoning_tokens=chunk.usage.completion_tokens_details.reasoning_tokens
                    )
                    if chunk.usage.completion_tokens_details
                    else None
                ),
            )
        else:
            # Accumulate across multiple inference calls
            self.accumulated_usage = OpenAIResponseUsage(
                input_tokens=self.accumulated_usage.input_tokens + chunk.usage.prompt_tokens,
                output_tokens=self.accumulated_usage.output_tokens + chunk.usage.completion_tokens,
                total_tokens=self.accumulated_usage.total_tokens + chunk.usage.total_tokens,
                # Use latest non-null details
                input_tokens_details=(
                    OpenAIResponseUsageInputTokensDetails(cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens)
                    if chunk.usage.prompt_tokens_details
                    else self.accumulated_usage.input_tokens_details
                ),
                output_tokens_details=(
                    OpenAIResponseUsageOutputTokensDetails(
                        reasoning_tokens=chunk.usage.completion_tokens_details.reasoning_tokens
                    )
                    if chunk.usage.completion_tokens_details
                    else self.accumulated_usage.output_tokens_details
                ),
            )

    async def _handle_reasoning_content_chunk(
        self,
        reasoning_content: str,
        reasoning_part_emitted: bool,
        reasoning_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Emit content_part.added event for first reasoning chunk
        if not reasoning_part_emitted:
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseContentPartAdded(
                content_index=reasoning_content_index,
                response_id=self.response_id,
                item_id=message_item_id,
                output_index=message_output_index,
                part=OpenAIResponseContentPartReasoningText(
                    text="",  # Will be filled incrementally via reasoning deltas
                ),
                sequence_number=self.sequence_number,
            )
        # Emit reasoning_text.delta event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseReasoningTextDelta(
            content_index=reasoning_content_index,
            delta=reasoning_content,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )

    async def _handle_refusal_content_chunk(
        self,
        refusal_content: str,
        refusal_part_emitted: bool,
        refusal_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Emit content_part.added event for first refusal chunk
        if not refusal_part_emitted:
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseContentPartAdded(
                content_index=refusal_content_index,
                response_id=self.response_id,
                item_id=message_item_id,
                output_index=message_output_index,
                part=OpenAIResponseContentPartRefusal(
                    refusal="",  # Will be filled incrementally via refusal deltas
                ),
                sequence_number=self.sequence_number,
            )
        # Emit refusal.delta event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseRefusalDelta(
            content_index=refusal_content_index,
            delta=refusal_content,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )

    async def _emit_reasoning_done_events(
        self,
        reasoning_text_accumulated: list[str],
        reasoning_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        final_reasoning_text = "".join(reasoning_text_accumulated)
        # Emit reasoning_text.done event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseReasoningTextDone(
            content_index=reasoning_content_index,
            text=final_reasoning_text,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )
        # Emit content_part.done for reasoning
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseContentPartDone(
            content_index=reasoning_content_index,
            response_id=self.response_id,
            item_id=message_item_id,
            output_index=message_output_index,
            part=OpenAIResponseContentPartReasoningText(
                text=final_reasoning_text,
            ),
            sequence_number=self.sequence_number,
        )

    async def _emit_refusal_done_events(
        self,
        refusal_text_accumulated: list[str],
        refusal_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        final_refusal_text = "".join(refusal_text_accumulated)
        # Emit refusal.done event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseRefusalDone(
            content_index=refusal_content_index,
            refusal=final_refusal_text,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )
        # Emit content_part.done for refusal
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseContentPartDone(
            content_index=refusal_content_index,
            response_id=self.response_id,
            item_id=message_item_id,
            output_index=message_output_index,
            part=OpenAIResponseContentPartRefusal(
                refusal=final_refusal_text,
            ),
            sequence_number=self.sequence_number,
        )

    async def _process_streaming_chunks(
        self, completion_result, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream | ChatCompletionResult]:
        """Process streaming chunks and emit events, returning completion data."""
        # Initialize result tracking
        chat_response_id = ""
        chat_response_content = []
        chat_response_tool_calls: dict[int, OpenAIChatCompletionToolCall] = {}
        chunk_created = 0
        chunk_model = ""
        chunk_finish_reason = ""

        # Create a placeholder message item for delta events
        message_item_id = f"msg_{uuid.uuid4()}"
        # Track tool call items for streaming events
        tool_call_item_ids: dict[int, str] = {}
        # Track content parts for streaming events
        message_item_added_emitted = False
        content_part_emitted = False
        reasoning_part_emitted = False
        refusal_part_emitted = False
        content_index = 0
        reasoning_content_index = 1  # reasoning is a separate content part
        refusal_content_index = 2  # refusal is a separate content part
        message_output_index = len(output_messages)
        reasoning_text_accumulated = []
        refusal_text_accumulated = []

        async for chunk in completion_result:
            chat_response_id = chunk.id
            chunk_created = chunk.created
            chunk_model = chunk.model

            # Accumulate usage from chunks (typically in final chunk with stream_options)
            self._accumulate_chunk_usage(chunk)

            # Track deltas for this specific chunk for guardrail validation
            chunk_events: list[OpenAIResponseObjectStream] = []

            for chunk_choice in chunk.choices:
                # Emit incremental text content as delta events
                if chunk_choice.delta.content:
                    # Emit output_item.added for the message on first content
                    if not message_item_added_emitted:
                        message_item_added_emitted = True
                        self.sequence_number += 1
                        message_item = OpenAIResponseMessage(
                            id=message_item_id,
                            content=[],
                            role="assistant",
                            status="in_progress",
                        )
                        yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                            response_id=self.response_id,
                            item=message_item,
                            output_index=message_output_index,
                            sequence_number=self.sequence_number,
                        )

                    # Emit content_part.added event for first text chunk
                    if not content_part_emitted:
                        content_part_emitted = True
                        self.sequence_number += 1
                        yield OpenAIResponseObjectStreamResponseContentPartAdded(
                            content_index=content_index,
                            response_id=self.response_id,
                            item_id=message_item_id,
                            output_index=message_output_index,
                            part=OpenAIResponseContentPartOutputText(
                                text="",  # Will be filled incrementally via text deltas
                            ),
                            sequence_number=self.sequence_number,
                        )
                    self.sequence_number += 1

                    text_delta_event = OpenAIResponseObjectStreamResponseOutputTextDelta(
                        content_index=content_index,
                        delta=chunk_choice.delta.content,
                        item_id=message_item_id,
                        output_index=message_output_index,
                        sequence_number=self.sequence_number,
                    )
                    # Buffer text delta events for guardrail check
                    if self.guardrail_ids:
                        chunk_events.append(text_delta_event)
                    else:
                        yield text_delta_event

                # Collect content for final response
                chat_response_content.append(chunk_choice.delta.content or "")
                if chunk_choice.finish_reason:
                    chunk_finish_reason = chunk_choice.finish_reason

                # Handle reasoning content if present (non-standard field for o1/o3 models)
                if hasattr(chunk_choice.delta, "reasoning_content") and chunk_choice.delta.reasoning_content:
                    async for event in self._handle_reasoning_content_chunk(
                        reasoning_content=chunk_choice.delta.reasoning_content,
                        reasoning_part_emitted=reasoning_part_emitted,
                        reasoning_content_index=reasoning_content_index,
                        message_item_id=message_item_id,
                        message_output_index=message_output_index,
                    ):
                        # Buffer reasoning events for guardrail check
                        if self.guardrail_ids:
                            chunk_events.append(event)
                        else:
                            yield event
                    reasoning_part_emitted = True
                    reasoning_text_accumulated.append(chunk_choice.delta.reasoning_content)

                # Handle refusal content if present
                if chunk_choice.delta.refusal:
                    async for event in self._handle_refusal_content_chunk(
                        refusal_content=chunk_choice.delta.refusal,
                        refusal_part_emitted=refusal_part_emitted,
                        refusal_content_index=refusal_content_index,
                        message_item_id=message_item_id,
                        message_output_index=message_output_index,
                    ):
                        yield event
                    refusal_part_emitted = True
                    refusal_text_accumulated.append(chunk_choice.delta.refusal)

                # Aggregate tool call arguments across chunks
                if chunk_choice.delta.tool_calls:
                    for tool_call in chunk_choice.delta.tool_calls:
                        response_tool_call = chat_response_tool_calls.get(tool_call.index, None)
                        # Create new tool call entry if this is the first chunk for this index
                        is_new_tool_call = response_tool_call is None
                        if is_new_tool_call:
                            tool_call_dict: dict[str, Any] = tool_call.model_dump()
                            tool_call_dict.pop("type", None)
                            response_tool_call = OpenAIChatCompletionToolCall(**tool_call_dict)
                            chat_response_tool_calls[tool_call.index] = response_tool_call

                            # Create item ID for this tool call for streaming events
                            tool_call_item_id = f"fc_{uuid.uuid4()}"
                            tool_call_item_ids[tool_call.index] = tool_call_item_id

                            # Emit output_item.added event for the new function call
                            self.sequence_number += 1
                            is_mcp_tool = tool_call.function.name and tool_call.function.name in self.mcp_tool_to_server
                            if not is_mcp_tool and tool_call.function.name not in ["web_search", "knowledge_search"]:
                                # for MCP tools (and even other non-function tools) we emit an output message item later
                                function_call_item = OpenAIResponseOutputMessageFunctionToolCall(
                                    arguments="",  # Will be filled incrementally via delta events
                                    call_id=tool_call.id or "",
                                    name=tool_call.function.name if tool_call.function else "",
                                    id=tool_call_item_id,
                                    status="in_progress",
                                )
                                yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                                    response_id=self.response_id,
                                    item=function_call_item,
                                    output_index=len(output_messages),
                                    sequence_number=self.sequence_number,
                                )

                        # Stream tool call arguments as they arrive (differentiate between MCP and function calls)
                        if tool_call.function and tool_call.function.arguments:
                            tool_call_item_id = tool_call_item_ids[tool_call.index]
                            self.sequence_number += 1

                            # Check if this is an MCP tool call
                            is_mcp_tool = tool_call.function.name and tool_call.function.name in self.mcp_tool_to_server
                            if is_mcp_tool:
                                # Emit MCP-specific argument delta event
                                yield OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta(
                                    delta=tool_call.function.arguments,
                                    item_id=tool_call_item_id,
                                    output_index=len(output_messages),
                                    sequence_number=self.sequence_number,
                                )
                            else:
                                # Emit function call argument delta event
                                yield OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta(
                                    delta=tool_call.function.arguments,
                                    item_id=tool_call_item_id,
                                    output_index=len(output_messages),
                                    sequence_number=self.sequence_number,
                                )

                            # Accumulate arguments for final response (only for subsequent chunks)
                            if not is_new_tool_call and response_tool_call is not None:
                                # Both should have functions since we're inside the tool_call.function check above
                                assert response_tool_call.function is not None
                                assert tool_call.function is not None
                                response_tool_call.function.arguments = (
                                    response_tool_call.function.arguments or ""
                                ) + tool_call.function.arguments

            # Output Safety Validation for this chunk
            if self.guardrail_ids:
                # Check guardrails on accumulated text so far
                accumulated_text = "".join(chat_response_content)
                violation_message = await run_guardrails(self.safety_api, accumulated_text, self.guardrail_ids)
                if violation_message:
                    logger.info(f"Output guardrail violation: {violation_message}")
                    chunk_events.clear()
                    yield await self._create_refusal_response(violation_message)
                    self.violation_detected = True
                    return
                else:
                    # No violation detected, emit all content events for this chunk
                    for event in chunk_events:
                        yield event

        # Emit arguments.done events for completed tool calls (differentiate between MCP and function calls)
        for tool_call_index in sorted(chat_response_tool_calls.keys()):
            tool_call = chat_response_tool_calls[tool_call_index]
            # Ensure that arguments, if sent back to the inference provider, are not None
            if tool_call.function:
                tool_call.function.arguments = tool_call.function.arguments or "{}"
            tool_call_item_id = tool_call_item_ids[tool_call_index]
            final_arguments: str = tool_call.function.arguments or "{}" if tool_call.function else "{}"
            func = chat_response_tool_calls[tool_call_index].function

            tool_call_name = func.name if func else ""

            # Check if this is an MCP tool call
            is_mcp_tool = tool_call_name and tool_call_name in self.mcp_tool_to_server
            self.sequence_number += 1
            done_event_cls = (
                OpenAIResponseObjectStreamResponseMcpCallArgumentsDone
                if is_mcp_tool
                else OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone
            )
            yield done_event_cls(
                arguments=final_arguments,
                item_id=tool_call_item_id,
                output_index=len(output_messages),
                sequence_number=self.sequence_number,
            )

        # Emit content_part.done event if text content was streamed (before content gets cleared)
        if content_part_emitted:
            final_text = "".join(chat_response_content)
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseContentPartDone(
                content_index=content_index,
                response_id=self.response_id,
                item_id=message_item_id,
                output_index=message_output_index,
                part=OpenAIResponseContentPartOutputText(
                    text=final_text,
                ),
                sequence_number=self.sequence_number,
            )

        # Emit reasoning done events if reasoning content was streamed
        if reasoning_part_emitted:
            async for event in self._emit_reasoning_done_events(
                reasoning_text_accumulated=reasoning_text_accumulated,
                reasoning_content_index=reasoning_content_index,
                message_item_id=message_item_id,
                message_output_index=message_output_index,
            ):
                yield event

        # Emit refusal done events if refusal content was streamed
        if refusal_part_emitted:
            async for event in self._emit_refusal_done_events(
                refusal_text_accumulated=refusal_text_accumulated,
                refusal_content_index=refusal_content_index,
                message_item_id=message_item_id,
                message_output_index=message_output_index,
            ):
                yield event

        # Clear content when there are tool calls (OpenAI spec behavior)
        if chat_response_tool_calls:
            chat_response_content = []

        # Emit output_item.done for message when we have content and no tool calls
        if message_item_added_emitted and not chat_response_tool_calls:
            content_parts = []
            if content_part_emitted:
                final_text = "".join(chat_response_content)
                content_parts.append(
                    OpenAIResponseOutputMessageContentOutputText(
                        text=final_text,
                        annotations=[],
                    )
                )

            self.sequence_number += 1
            message_item = OpenAIResponseMessage(
                id=message_item_id,
                content=content_parts,
                role="assistant",
                status="completed",
            )
            yield OpenAIResponseObjectStreamResponseOutputItemDone(
                response_id=self.response_id,
                item=message_item,
                output_index=message_output_index,
                sequence_number=self.sequence_number,
            )

        yield ChatCompletionResult(
            response_id=chat_response_id,
            content=chat_response_content,
            tool_calls=chat_response_tool_calls,
            created=chunk_created,
            model=chunk_model,
            finish_reason=chunk_finish_reason,
            message_item_id=message_item_id,
            tool_call_item_ids=tool_call_item_ids,
            content_part_emitted=content_part_emitted,
        )

    def _build_chat_completion(self, result: ChatCompletionResult) -> OpenAIChatCompletion:
        """Build OpenAIChatCompletion from ChatCompletionResult."""
        # Convert collected chunks to complete response
        if result.tool_calls:
            tool_calls = [result.tool_calls[i] for i in sorted(result.tool_calls.keys())]
        else:
            tool_calls = None

        assistant_message = OpenAIAssistantMessageParam(
            content=result.content_text,
            tool_calls=tool_calls,
        )
        return OpenAIChatCompletion(
            id=result.response_id,
            choices=[
                OpenAIChoice(
                    message=assistant_message,
                    finish_reason=result.finish_reason,
                    index=0,
                )
            ],
            created=result.created,
            model=result.model,
        )

    async def _coordinate_tool_execution(
        self,
        function_tool_calls: list,
        non_function_tool_calls: list,
        completion_result_data: ChatCompletionResult,
        output_messages: list[OpenAIResponseOutput],
        next_turn_messages: list,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Coordinate execution of both function and non-function tool calls."""
        # Execute non-function tool calls
        for tool_call in non_function_tool_calls:
            # Check if total calls made to built-in and mcp tools exceed max_tool_calls
            if self.max_tool_calls is not None and self.accumulated_builtin_tool_calls >= self.max_tool_calls:
                logger.info(f"Ignoring built-in and mcp tool call since reached the limit of {self.max_tool_calls=}.")
                break

            # Find the item_id for this tool call
            matching_item_id = None
            for index, item_id in completion_result_data.tool_call_item_ids.items():
                response_tool_call = completion_result_data.tool_calls.get(index)
                if response_tool_call and response_tool_call.id == tool_call.id:
                    matching_item_id = item_id
                    break

            # Use a fallback item_id if not found
            if not matching_item_id:
                matching_item_id = f"tc_{uuid.uuid4()}"

            self.sequence_number += 1
            if tool_call.function.name and tool_call.function.name in self.mcp_tool_to_server:
                item: OpenAIResponseOutput = OpenAIResponseOutputMessageMCPCall(
                    arguments="",
                    name=tool_call.function.name,
                    id=matching_item_id,
                    server_label=self.mcp_tool_to_server[tool_call.function.name].server_label,
                )
            elif tool_call.function.name == "web_search":
                item = OpenAIResponseOutputMessageWebSearchToolCall(
                    id=matching_item_id,
                    status="in_progress",
                )
            elif tool_call.function.name == "knowledge_search":
                item = OpenAIResponseOutputMessageFileSearchToolCall(
                    id=matching_item_id,
                    status="in_progress",
                    queries=[tool_call.function.arguments or ""],
                )
            else:
                raise ValueError(f"Unsupported tool call: {tool_call.function.name}")

            yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                response_id=self.response_id,
                item=item,
                output_index=len(output_messages),
                sequence_number=self.sequence_number,
            )

            # Execute tool call with streaming
            tool_call_log = None
            tool_response_message = None
            async for result in self.tool_executor.execute_tool_call(
                tool_call,
                self.ctx,
                self.sequence_number,
                len(output_messages),
                matching_item_id,
                self.mcp_tool_to_server,
            ):
                if result.stream_event:
                    # Forward streaming events
                    self.sequence_number = result.sequence_number
                    yield result.stream_event

                if result.final_output_message is not None:
                    tool_call_log = result.final_output_message
                    tool_response_message = result.final_input_message
                    self.sequence_number = result.sequence_number
                    if result.citation_files:
                        self.citation_files.update(result.citation_files)

            if tool_call_log:
                output_messages.append(tool_call_log)

                # Emit output_item.done event for completed non-function tool call
                if matching_item_id:
                    self.sequence_number += 1
                    yield OpenAIResponseObjectStreamResponseOutputItemDone(
                        response_id=self.response_id,
                        item=tool_call_log,
                        output_index=len(output_messages) - 1,
                        sequence_number=self.sequence_number,
                    )

            if tool_response_message:
                next_turn_messages.append(tool_response_message)

            # Track number of calls made to built-in and mcp tools
            self.accumulated_builtin_tool_calls += 1

        # Execute function tool calls (client-side)
        for tool_call in function_tool_calls:
            # Find the item_id for this tool call from our tracking dictionary
            matching_item_id = None
            for index, item_id in completion_result_data.tool_call_item_ids.items():
                response_tool_call = completion_result_data.tool_calls.get(index)
                if response_tool_call and response_tool_call.id == tool_call.id:
                    matching_item_id = item_id
                    break

            # Use existing item_id or create new one if not found
            final_item_id = matching_item_id or f"fc_{uuid.uuid4()}"

            function_call_item = OpenAIResponseOutputMessageFunctionToolCall(
                arguments=tool_call.function.arguments or "",
                call_id=tool_call.id,
                name=tool_call.function.name or "",
                id=final_item_id,
                status="completed",
            )
            output_messages.append(function_call_item)

            # Emit output_item.done event for completed function call
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseOutputItemDone(
                response_id=self.response_id,
                item=function_call_item,
                output_index=len(output_messages) - 1,
                sequence_number=self.sequence_number,
            )

    async def _process_new_tools(
        self, tools: list[OpenAIResponseInputTool], output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Process all tools and emit appropriate streaming events."""
        from openai.types.chat import ChatCompletionToolParam

        from llama_stack.apis.tools import ToolDef
        from llama_stack.models.llama.datatypes import ToolDefinition
        from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool

        def make_openai_tool(tool_name: str, tool: ToolDef) -> ChatCompletionToolParam:
            tool_def = ToolDefinition(
                tool_name=tool_name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            return convert_tooldef_to_openai_tool(tool_def)  # type: ignore[return-value]  # Returns dict but ChatCompletionToolParam expects TypedDict

        # Initialize chat_tools if not already set
        if self.ctx.chat_tools is None:
            self.ctx.chat_tools = []

        for input_tool in tools:
            if input_tool.type == "function":
                self.ctx.chat_tools.append(ChatCompletionToolParam(type="function", function=input_tool.model_dump()))  # type: ignore[typeddict-item,arg-type]  # Dict compatible with FunctionDefinition
            elif input_tool.type in WebSearchToolTypes:
                tool_name = "web_search"
                # Need to access tool_groups_api from tool_executor
                tool = await self.tool_executor.tool_groups_api.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                self.ctx.chat_tools.append(make_openai_tool(tool_name, tool))
            elif input_tool.type == "file_search":
                tool_name = "knowledge_search"
                tool = await self.tool_executor.tool_groups_api.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                self.ctx.chat_tools.append(make_openai_tool(tool_name, tool))
            elif input_tool.type == "mcp":
                async for stream_event in self._process_mcp_tool(input_tool, output_messages):
                    yield stream_event
            else:
                raise ValueError(f"Llama Stack OpenAI Responses does not yet support tool type: {input_tool.type}")

    async def _process_mcp_tool(
        self, mcp_tool: OpenAIResponseInputToolMCP, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Process an MCP tool configuration and emit appropriate streaming events."""
        from llama_stack.providers.utils.tools.mcp import list_mcp_tools

        # Emit mcp_list_tools.in_progress
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseMcpListToolsInProgress(
            sequence_number=self.sequence_number,
        )
        try:
            # Parse allowed/never allowed tools
            always_allowed = None
            never_allowed = None
            if mcp_tool.allowed_tools:
                if isinstance(mcp_tool.allowed_tools, list):
                    always_allowed = mcp_tool.allowed_tools
                elif isinstance(mcp_tool.allowed_tools, AllowedToolsFilter):
                    # AllowedToolsFilter only has tool_names field (not allowed/disallowed)
                    always_allowed = mcp_tool.allowed_tools.tool_names

            # Call list_mcp_tools
            tool_defs = None
            list_id = f"mcp_list_{uuid.uuid4()}"
            attributes = {
                "server_label": mcp_tool.server_label,
                "server_url": mcp_tool.server_url,
                "mcp_list_tools_id": list_id,
            }
            async with tracing.span("list_mcp_tools", attributes):
                tool_defs = await list_mcp_tools(
                    endpoint=mcp_tool.server_url,
                    headers=mcp_tool.headers or {},
                )

            # Create the MCP list tools message
            mcp_list_message = OpenAIResponseOutputMessageMCPListTools(
                id=list_id,
                server_label=mcp_tool.server_label,
                tools=[],
            )

            # Process tools and update context
            for t in tool_defs.data:
                if never_allowed and t.name in never_allowed:
                    continue
                if not always_allowed or t.name in always_allowed:
                    # Add to chat tools for inference
                    openai_tool = convert_tooldef_to_chat_tool(t)
                    if self.ctx.chat_tools is None:
                        self.ctx.chat_tools = []
                    self.ctx.chat_tools.append(openai_tool)  # type: ignore[arg-type]  # Returns dict but ChatCompletionToolParam expects TypedDict

                    # Add to MCP tool mapping
                    if t.name in self.mcp_tool_to_server:
                        raise ValueError(f"Duplicate tool name {t.name} found for server {mcp_tool.server_label}")
                    self.mcp_tool_to_server[t.name] = mcp_tool

                    # Add to MCP list message
                    mcp_list_message.tools.append(
                        MCPListToolsTool(
                            name=t.name,
                            description=t.description,
                            input_schema=t.input_schema
                            or {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        )
                    )
            async for stream_event in self._add_mcp_list_tools(mcp_list_message, output_messages):
                yield stream_event

        except Exception as e:
            # TODO: Emit mcp_list_tools.failed event if needed
            logger.exception(f"Failed to list MCP tools from {mcp_tool.server_url}: {e}")
            raise

    async def _process_tools(
        self, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Handle all mcp tool lists from previous response that are still valid:
        # tool_context can be None when no tools are provided in the response request
        if self.ctx.tool_context:
            for tool in self.ctx.tool_context.previous_tool_listings:
                async for evt in self._reuse_mcp_list_tools(tool, output_messages):
                    yield evt
            # Process all remaining tools (including MCP tools) and emit streaming events
            if self.ctx.tool_context.tools_to_process:
                async for stream_event in self._process_new_tools(
                    self.ctx.tool_context.tools_to_process, output_messages
                ):
                    yield stream_event

    def _approval_required(self, tool_name: str) -> bool:
        if tool_name not in self.mcp_tool_to_server:
            return False
        mcp_server = self.mcp_tool_to_server[tool_name]
        if mcp_server.require_approval == "always":
            return True
        if mcp_server.require_approval == "never":
            return False
        if isinstance(mcp_server, ApprovalFilter):
            if tool_name in mcp_server.always:
                return True
            if tool_name in mcp_server.never:
                return False
        return True

    async def _add_mcp_approval_request(
        self, tool_name: str, arguments: str, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        mcp_server = self.mcp_tool_to_server[tool_name]
        mcp_approval_request = OpenAIResponseMCPApprovalRequest(
            arguments=arguments,
            id=f"approval_{uuid.uuid4()}",
            name=tool_name,
            server_label=mcp_server.server_label,
        )
        output_messages.append(mcp_approval_request)

        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemAdded(
            response_id=self.response_id,
            item=mcp_approval_request,
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemDone(
            response_id=self.response_id,
            item=mcp_approval_request,
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )

    async def _add_mcp_list_tools(
        self, mcp_list_message: OpenAIResponseOutputMessageMCPListTools, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Add the MCP list message to output
        output_messages.append(mcp_list_message)

        # Emit output_item.added for the MCP list tools message
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemAdded(
            response_id=self.response_id,
            item=OpenAIResponseOutputMessageMCPListTools(
                id=mcp_list_message.id,
                server_label=mcp_list_message.server_label,
                tools=[],
            ),
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )
        # Emit mcp_list_tools.completed
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseMcpListToolsCompleted(
            sequence_number=self.sequence_number,
        )

        # Emit output_item.done for the MCP list tools message
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemDone(
            response_id=self.response_id,
            item=mcp_list_message,
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )

    async def _reuse_mcp_list_tools(
        self, original: OpenAIResponseOutputMessageMCPListTools, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        for t in original.tools:
            from llama_stack.models.llama.datatypes import ToolDefinition
            from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool

            # convert from input_schema to map of ToolParamDefinitions...
            tool_def = ToolDefinition(
                tool_name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            # ...then can convert that to openai completions tool
            openai_tool = convert_tooldef_to_openai_tool(tool_def)
            if self.ctx.chat_tools is None:
                self.ctx.chat_tools = []
            self.ctx.chat_tools.append(openai_tool)  # type: ignore[arg-type]  # Returns dict but ChatCompletionToolParam expects TypedDict

        mcp_list_message = OpenAIResponseOutputMessageMCPListTools(
            id=f"mcp_list_{uuid.uuid4()}",
            server_label=original.server_label,
            tools=original.tools,
        )

        async for stream_event in self._add_mcp_list_tools(mcp_list_message, output_messages):
            yield stream_event
