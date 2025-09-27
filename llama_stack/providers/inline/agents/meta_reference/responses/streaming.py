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
    MCPListToolsTool,
    OpenAIResponseContentPartOutputText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolMCP,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseContentPartAdded,
    OpenAIResponseObjectStreamResponseContentPartDone,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDone,
    OpenAIResponseObjectStreamResponseMcpListToolsCompleted,
    OpenAIResponseObjectStreamResponseMcpListToolsInProgress,
    OpenAIResponseObjectStreamResponseOutputItemAdded,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseText,
    WebSearchToolTypes,
)
from llama_stack.apis.inference import (
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionToolCall,
    OpenAIChoice,
)
from llama_stack.log import get_logger

from .types import ChatCompletionContext, ChatCompletionResult
from .utils import convert_chat_choice_to_response_message, is_function_tool_call

logger = get_logger(name=__name__, category="agents::meta_reference")


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
    ):
        self.inference_api = inference_api
        self.ctx = ctx
        self.response_id = response_id
        self.created_at = created_at
        self.text = text
        self.max_infer_iters = max_infer_iters
        self.tool_executor = tool_executor
        self.sequence_number = 0
        # Store MCP tool mapping that gets built during tool processing
        self.mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] = {}

    async def create_response(self) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Initialize output messages
        output_messages: list[OpenAIResponseOutput] = []
        # Create initial response and emit response.created immediately
        initial_response = OpenAIResponseObject(
            created_at=self.created_at,
            id=self.response_id,
            model=self.ctx.model,
            object="response",
            status="in_progress",
            output=output_messages.copy(),
            text=self.text,
        )

        yield OpenAIResponseObjectStreamResponseCreated(response=initial_response)

        # Process all tools (including MCP tools) and emit streaming events
        if self.ctx.response_tools:
            async for stream_event in self._process_tools(self.ctx.response_tools, output_messages):
                yield stream_event

        n_iter = 0
        messages = self.ctx.messages.copy()

        while True:
            # Only forward response_format when it's a JSON schema; omit "text"/"json_object" to prevent
            # OpenAI-compatible providers (e.g., vLLM) from enabling structured output without a schema.
            safe_response_format = self.ctx.response_format
            try:
                if safe_response_format is not None and getattr(safe_response_format, "type", None) != "json_schema":
                    safe_response_format = None
            except Exception:
                safe_response_format = None

            completion_result = await self.inference_api.openai_chat_completion(
                model=self.ctx.model,
                messages=messages,
                tools=self.ctx.chat_tools,
                stream=True,
                temperature=self.ctx.temperature,
                response_format=safe_response_format,
            )

            # Process streaming chunks and build complete response
            completion_result_data = None
            async for stream_event_or_result in self._process_streaming_chunks(completion_result, output_messages):
                if isinstance(stream_event_or_result, ChatCompletionResult):
                    completion_result_data = stream_event_or_result
                else:
                    yield stream_event_or_result
            if not completion_result_data:
                raise ValueError("Streaming chunk processor failed to return completion data")
            current_response = self._build_chat_completion(completion_result_data)

            function_tool_calls, non_function_tool_calls, next_turn_messages = self._separate_tool_calls(
                current_response, messages
            )

            # Handle choices with no tool calls
            for choice in current_response.choices:
                if not (choice.message.tool_calls and self.ctx.response_tools):
                    output_messages.append(await convert_chat_choice_to_response_message(choice))

            # Execute tool calls and coordinate results
            async for stream_event in self._coordinate_tool_execution(
                function_tool_calls,
                non_function_tool_calls,
                completion_result_data,
                output_messages,
                next_turn_messages,
            ):
                yield stream_event

            if not function_tool_calls and not non_function_tool_calls:
                break

            if function_tool_calls:
                logger.info("Exiting inference loop since there is a function (client-side) tool call")
                break

            n_iter += 1
            if n_iter >= self.max_infer_iters:
                logger.info(f"Exiting inference loop since iteration count({n_iter}) exceeds {self.max_infer_iters=}")
                break

            messages = next_turn_messages

        # Create final response
        final_response = OpenAIResponseObject(
            created_at=self.created_at,
            id=self.response_id,
            model=self.ctx.model,
            object="response",
            status="completed",
            text=self.text,
            output=output_messages,
        )

        # Emit response.completed
        yield OpenAIResponseObjectStreamResponseCompleted(response=final_response)

    def _separate_tool_calls(self, current_response, messages) -> tuple[list, list, list]:
        """Separate tool calls into function and non-function categories."""
        function_tool_calls = []
        non_function_tool_calls = []
        next_turn_messages = messages.copy()

        for choice in current_response.choices:
            next_turn_messages.append(choice.message)

            if choice.message.tool_calls and self.ctx.response_tools:
                for tool_call in choice.message.tool_calls:
                    if is_function_tool_call(tool_call, self.ctx.response_tools):
                        function_tool_calls.append(tool_call)
                    else:
                        non_function_tool_calls.append(tool_call)

        return function_tool_calls, non_function_tool_calls, next_turn_messages

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
        content_part_emitted = False

        async for chunk in completion_result:
            chat_response_id = chunk.id
            chunk_created = chunk.created
            chunk_model = chunk.model
            for chunk_choice in chunk.choices:
                # Emit incremental text content as delta events
                if chunk_choice.delta.content:
                    # Emit content_part.added event for first text chunk
                    if not content_part_emitted:
                        content_part_emitted = True
                        self.sequence_number += 1
                        yield OpenAIResponseObjectStreamResponseContentPartAdded(
                            response_id=self.response_id,
                            item_id=message_item_id,
                            part=OpenAIResponseContentPartOutputText(
                                text="",  # Will be filled incrementally via text deltas
                            ),
                            sequence_number=self.sequence_number,
                        )
                    self.sequence_number += 1
                    yield OpenAIResponseObjectStreamResponseOutputTextDelta(
                        content_index=0,
                        delta=chunk_choice.delta.content,
                        item_id=message_item_id,
                        output_index=0,
                        sequence_number=self.sequence_number,
                    )

                # Collect content for final response
                chat_response_content.append(chunk_choice.delta.content or "")
                if chunk_choice.finish_reason:
                    chunk_finish_reason = chunk_choice.finish_reason

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
                            if not is_new_tool_call:
                                response_tool_call.function.arguments = (
                                    response_tool_call.function.arguments or ""
                                ) + tool_call.function.arguments

        # Emit arguments.done events for completed tool calls (differentiate between MCP and function calls)
        for tool_call_index in sorted(chat_response_tool_calls.keys()):
            tool_call_item_id = tool_call_item_ids[tool_call_index]
            final_arguments = chat_response_tool_calls[tool_call_index].function.arguments or ""
            tool_call_name = chat_response_tool_calls[tool_call_index].function.name

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
                response_id=self.response_id,
                item_id=message_item_id,
                part=OpenAIResponseContentPartOutputText(
                    text=final_text,
                ),
                sequence_number=self.sequence_number,
            )

        # Clear content when there are tool calls (OpenAI spec behavior)
        if chat_response_tool_calls:
            chat_response_content = []

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

    async def _process_tools(
        self, tools: list[OpenAIResponseInputTool], output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Process all tools and emit appropriate streaming events."""
        from openai.types.chat import ChatCompletionToolParam

        from llama_stack.apis.tools import Tool
        from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
        from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool

        def make_openai_tool(tool_name: str, tool: Tool) -> ChatCompletionToolParam:
            tool_def = ToolDefinition(
                tool_name=tool_name,
                description=tool.description,
                parameters={
                    param.name: ToolParamDefinition(
                        param_type=param.parameter_type,
                        description=param.description,
                        required=param.required,
                        default=param.default,
                    )
                    for param in tool.parameters
                },
            )
            return convert_tooldef_to_openai_tool(tool_def)

        # Initialize chat_tools if not already set
        if self.ctx.chat_tools is None:
            self.ctx.chat_tools = []

        for input_tool in tools:
            if input_tool.type == "function":
                self.ctx.chat_tools.append(ChatCompletionToolParam(type="function", function=input_tool.model_dump()))
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
                    always_allowed = mcp_tool.allowed_tools.always
                    never_allowed = mcp_tool.allowed_tools.never

            # Call list_mcp_tools
            tool_defs = await list_mcp_tools(
                endpoint=mcp_tool.server_url,
                headers=mcp_tool.headers or {},
            )

            # Create the MCP list tools message
            mcp_list_message = OpenAIResponseOutputMessageMCPListTools(
                id=f"mcp_list_{uuid.uuid4()}",
                server_label=mcp_tool.server_label,
                tools=[],
            )

            # Process tools and update context
            for t in tool_defs.data:
                if never_allowed and t.name in never_allowed:
                    continue
                if not always_allowed or t.name in always_allowed:
                    # Add to chat tools for inference
                    from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
                    from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool

                    tool_def = ToolDefinition(
                        tool_name=t.name,
                        description=t.description,
                        parameters={
                            param.name: ToolParamDefinition(
                                param_type=param.parameter_type,
                                description=param.description,
                                required=param.required,
                                default=param.default,
                            )
                            for param in t.parameters
                        },
                    )
                    openai_tool = convert_tooldef_to_openai_tool(tool_def)
                    if self.ctx.chat_tools is None:
                        self.ctx.chat_tools = []
                    self.ctx.chat_tools.append(openai_tool)

                    # Add to MCP tool mapping
                    if t.name in self.mcp_tool_to_server:
                        raise ValueError(f"Duplicate tool name {t.name} found for server {mcp_tool.server_label}")
                    self.mcp_tool_to_server[t.name] = mcp_tool

                    # Add to MCP list message
                    mcp_list_message.tools.append(
                        MCPListToolsTool(
                            name=t.name,
                            description=t.description,
                            input_schema={
                                "type": "object",
                                "properties": {
                                    p.name: {
                                        "type": p.parameter_type,
                                        "description": p.description,
                                    }
                                    for p in t.parameters
                                },
                                "required": [p.name for p in t.parameters if p.required],
                            },
                        )
                    )

            # Add the MCP list message to output
            output_messages.append(mcp_list_message)

            # Emit output_item.added for the MCP list tools message
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                response_id=self.response_id,
                item=mcp_list_message,
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

        except Exception as e:
            # TODO: Emit mcp_list_tools.failed event if needed
            logger.exception(f"Failed to list MCP tools from {mcp_tool.server_url}: {e}")
            raise
