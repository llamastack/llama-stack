# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import re
import time
import uuid
from collections.abc import AsyncIterator

from pydantic import BaseModel, TypeAdapter

from llama_stack.log import get_logger
from llama_stack.providers.utils.responses.responses_store import (
    ResponsesStore,
    _OpenAIResponseObjectWithInputAndMessages,
)
from llama_stack.providers.utils.tools.mcp import MCPSessionManager
from llama_stack_api import (
    ConversationItem,
    Conversations,
    Files,
    Inference,
    InvalidConversationIdError,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIChatCompletionContentPartParam,
    OpenAIDeleteResponseObject,
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContentFile,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
    Order,
    Prompts,
    ResponseGuardrailSpec,
    ResponseItemInclude,
    Safety,
    ToolGroups,
    ToolRuntime,
    VectorIO,
)

from .streaming import StreamingResponseOrchestrator
from .tool_executor import ToolExecutor
from .types import ChatCompletionContext, ToolContext
from .utils import (
    convert_response_content_to_chat_content,
    convert_response_input_to_chat_messages,
    convert_response_text_to_chat_response_format,
    extract_guardrail_ids,
)

logger = get_logger(name=__name__, category="openai_responses")


class OpenAIResponsePreviousResponseWithInputItems(BaseModel):
    input_items: ListOpenAIResponseInputItem
    response: OpenAIResponseObject


class OpenAIResponsesImpl:
    def __init__(
        self,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
        responses_store: ResponsesStore,
        vector_io_api: VectorIO,  # VectorIO
        safety_api: Safety | None,
        conversations_api: Conversations,
        prompts_api: Prompts,
        files_api: Files,
        vector_stores_config=None,
    ):
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api
        self.responses_store = responses_store
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.conversations_api = conversations_api
        self.tool_executor = ToolExecutor(
            tool_groups_api=tool_groups_api,
            tool_runtime_api=tool_runtime_api,
            vector_io_api=vector_io_api,
            vector_stores_config=vector_stores_config,
        )
        self.prompts_api = prompts_api
        self.files_api = files_api

    async def _prepend_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        previous_response: _OpenAIResponseObjectWithInputAndMessages,
    ):
        # Convert Sequence to list for mutation
        new_input_items = list(previous_response.input)
        new_input_items.extend(previous_response.output)

        if isinstance(input, str):
            new_input_items.append(OpenAIResponseMessage(content=input, role="user"))
        else:
            new_input_items.extend(input)

        return new_input_items

    async def _process_input_with_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        tools: list[OpenAIResponseInputTool] | None,
        previous_response_id: str | None,
        conversation: str | None,
    ) -> tuple[str | list[OpenAIResponseInput], list[OpenAIMessageParam], ToolContext]:
        """Process input with optional previous response context.

        Returns:
            tuple: (all_input for storage, messages for chat completion, tool context)
        """
        tool_context = ToolContext(tools)
        if previous_response_id:
            previous_response: _OpenAIResponseObjectWithInputAndMessages = (
                await self.responses_store.get_response_object(previous_response_id)
            )
            all_input = await self._prepend_previous_response(input, previous_response)

            if previous_response.messages:
                # Use stored messages directly and convert only new input
                message_adapter = TypeAdapter(list[OpenAIMessageParam])
                messages = message_adapter.validate_python(previous_response.messages)
                new_messages = await convert_response_input_to_chat_messages(
                    input, previous_messages=messages, files_api=self.files_api
                )
                messages.extend(new_messages)
            else:
                # Backward compatibility: reconstruct from inputs
                messages = await convert_response_input_to_chat_messages(all_input, files_api=self.files_api)

            tool_context.recover_tools_from_previous_response(previous_response)
        elif conversation is not None:
            conversation_items = await self.conversations_api.list_items(conversation, order="asc")

            # Use stored messages as source of truth (like previous_response.messages)
            stored_messages = await self.responses_store.get_conversation_messages(conversation)

            all_input = input
            if not conversation_items.data:
                # First turn - just convert the new input
                messages = await convert_response_input_to_chat_messages(input, files_api=self.files_api)
            else:
                if not stored_messages:
                    all_input = conversation_items.data
                    if isinstance(input, str):
                        all_input.append(
                            OpenAIResponseMessage(
                                role="user", content=[OpenAIResponseInputMessageContentText(text=input)]
                            )
                        )
                    else:
                        all_input.extend(input)
                else:
                    all_input = input

                messages = stored_messages or []
                new_messages = await convert_response_input_to_chat_messages(
                    all_input, previous_messages=messages, files_api=self.files_api
                )
                messages.extend(new_messages)
        else:
            all_input = input
            messages = await convert_response_input_to_chat_messages(all_input, files_api=self.files_api)

        return all_input, messages, tool_context

    async def _prepend_prompt(
        self,
        messages: list[OpenAIMessageParam],
        openai_response_prompt: OpenAIResponsePrompt | None,
    ) -> None:
        """Prepend prompt template to messages, resolving text/image/file variables.

        :param messages: List of OpenAIMessageParam objects
        :param openai_response_prompt: (Optional) OpenAIResponsePrompt object with variables
        :returns: string of utf-8 characters
        """
        if not openai_response_prompt or not openai_response_prompt.id:
            return

        prompt_version = int(openai_response_prompt.version) if openai_response_prompt.version else None
        cur_prompt = await self.prompts_api.get_prompt(openai_response_prompt.id, prompt_version)

        if not cur_prompt or not cur_prompt.prompt:
            return

        cur_prompt_text = cur_prompt.prompt
        cur_prompt_variables = cur_prompt.variables

        if not openai_response_prompt.variables:
            messages.insert(0, OpenAISystemMessageParam(content=cur_prompt_text))
            return

        # Validate that all provided variables exist in the prompt
        for name in openai_response_prompt.variables.keys():
            if name not in cur_prompt_variables:
                raise ValueError(f"Variable {name} not found in prompt {openai_response_prompt.id}")

        # Separate text and media variables
        text_substitutions = {}
        media_content_parts: list[OpenAIChatCompletionContentPartParam] = []

        for name, value in openai_response_prompt.variables.items():
            # Text variable found
            if isinstance(value, OpenAIResponseInputMessageContentText):
                text_substitutions[name] = value.text

            # Media variable found
            elif isinstance(value, OpenAIResponseInputMessageContentImage | OpenAIResponseInputMessageContentFile):
                converted_parts = await convert_response_content_to_chat_content([value], files_api=self.files_api)
                if isinstance(converted_parts, list):
                    media_content_parts.extend(converted_parts)

                # Eg: {{product_photo}} becomes "[Image: product_photo]"
                # This gives the model textual context about what media exists in the prompt
                var_type = value.type.replace("input_", "").replace("_", " ").title()
                text_substitutions[name] = f"[{var_type}: {name}]"

        def replace_variable(match: re.Match[str]) -> str:
            var_name = match.group(1).strip()
            return str(text_substitutions.get(var_name, match.group(0)))

        pattern = r"\{\{\s*(\w+)\s*\}\}"
        processed_prompt_text = re.sub(pattern, replace_variable, cur_prompt_text)

        # Insert system message with resolved text
        messages.insert(0, OpenAISystemMessageParam(content=processed_prompt_text))

        # If we have media, create a new user message because allows to ingest images and files
        if media_content_parts:
            messages.append(OpenAIUserMessageParam(content=media_content_parts))

    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        response_with_input = await self.responses_store.get_response_object(response_id)
        return response_with_input.to_response_object()

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        return await self.responses_store.list_responses(after, limit, model, order)

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[ResponseItemInclude] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items for a given OpenAI response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        :returns: An ListOpenAIResponseInputItem.
        """
        return await self.responses_store.list_response_input_items(response_id, after, before, include, limit, order)

    async def _store_response(
        self,
        response: OpenAIResponseObject,
        input: str | list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
    ) -> None:
        new_input_id = f"msg_{uuid.uuid4()}"
        # Type input_items_data as the full OpenAIResponseInput union to avoid list invariance issues
        input_items_data: list[OpenAIResponseInput] = []

        if isinstance(input, str):
            # synthesize a message from the input string
            input_content = OpenAIResponseInputMessageContentText(text=input)
            input_content_item = OpenAIResponseMessage(
                role="user",
                content=[input_content],
                id=new_input_id,
            )
            input_items_data = [input_content_item]
        else:
            # we already have a list of messages
            for input_item in input:
                if isinstance(input_item, OpenAIResponseMessage):
                    # These may or may not already have an id, so dump to dict, check for id, and add if missing
                    input_item_dict = input_item.model_dump()
                    if "id" not in input_item_dict:
                        input_item_dict["id"] = new_input_id
                    input_items_data.append(OpenAIResponseMessage(**input_item_dict))
                else:
                    input_items_data.append(input_item)

        await self.responses_store.store_response_object(
            response_object=response,
            input=input_items_data,
            messages=messages,
        )

    def _prepare_input_items_for_storage(
        self,
        input: str | list[OpenAIResponseInput],
    ) -> list[OpenAIResponseInput]:
        """Prepare input items for storage, adding IDs where needed.

        This method is called once at the start of streaming to prepare input items
        that will be reused across multiple persistence calls during streaming.
        """
        new_input_id = f"msg_{uuid.uuid4()}"
        input_items_data: list[OpenAIResponseInput] = []

        if isinstance(input, str):
            input_content = OpenAIResponseInputMessageContentText(text=input)
            input_content_item = OpenAIResponseMessage(
                role="user",
                content=[input_content],
                id=new_input_id,
            )
            input_items_data = [input_content_item]
        else:
            for input_item in input:
                if isinstance(input_item, OpenAIResponseMessage):
                    input_item_dict = input_item.model_dump()
                    if "id" not in input_item_dict:
                        input_item_dict["id"] = new_input_id
                    input_items_data.append(OpenAIResponseMessage(**input_item_dict))
                else:
                    input_items_data.append(input_item)

        return input_items_data

    async def _persist_streaming_state(
        self,
        stream_chunk: OpenAIResponseObjectStream,
        orchestrator,
        input_items: list[OpenAIResponseInput],
        output_items: list,
    ) -> None:
        """Persist response state at significant streaming events.

        This enables clients to poll GET /v1/responses/{response_id} during streaming
        to see in-progress turn state instead of empty results.

        Persistence occurs at:
        - response.in_progress: Initial INSERT with empty output
        - response.output_item.done: UPDATE with accumulated output items
        - response.completed/response.incomplete: Final UPDATE with complete state
        - response.failed: UPDATE with error state

        :param stream_chunk: The current streaming event.
        :param orchestrator: The streaming orchestrator (for snapshotting response).
        :param input_items: Pre-prepared input items for storage.
        :param output_items: Accumulated output items so far.
        """
        try:
            match stream_chunk.type:
                case "response.in_progress":
                    # Initial persistence when response starts
                    in_progress_response = stream_chunk.response
                    await self.responses_store.upsert_response_object(
                        response_object=in_progress_response,
                        input=input_items,
                        messages=[],
                    )

                case "response.output_item.done":
                    # Incremental update when an output item completes (tool call, message)
                    current_snapshot = orchestrator._snapshot_response(
                        status="in_progress",
                        outputs=output_items,
                    )
                    # Get current messages (filter out system messages)
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages or orchestrator.ctx.messages,
                        )
                    )
                    await self.responses_store.upsert_response_object(
                        response_object=current_snapshot,
                        input=input_items,
                        messages=messages_to_store,
                    )

                case "response.completed" | "response.incomplete":
                    # Final persistence when response finishes
                    final_response = stream_chunk.response
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages,
                        )
                    )
                    await self.responses_store.upsert_response_object(
                        response_object=final_response,
                        input=input_items,
                        messages=messages_to_store,
                    )

                case "response.failed":
                    # Persist failed state so GET shows error
                    failed_response = stream_chunk.response
                    # Preserve any accumulated non-system messages for failed responses
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages or orchestrator.ctx.messages,
                        )
                    )
                    await self.responses_store.upsert_response_object(
                        response_object=failed_response,
                        input=input_items,
                        messages=messages_to_store,
                    )
        except Exception as e:
            # Best-effort persistence: log error but don't fail the stream
            logger.warning(f"Failed to persist streaming state for {stream_chunk.type}: {e}")

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
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[ResponseItemInclude] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[str | ResponseGuardrailSpec] | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        metadata: dict[str, str] | None = None,
    ):
        stream = bool(stream)
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")) if text is None else text

        # Validate MCP tools: ensure Authorization header is not passed via headers dict
        if tools:
            from llama_stack_api.openai_responses import OpenAIResponseInputToolMCP

            for tool in tools:
                if isinstance(tool, OpenAIResponseInputToolMCP) and tool.headers:
                    for key in tool.headers.keys():
                        if key.lower() == "authorization":
                            raise ValueError(
                                "Authorization header cannot be passed via 'headers'. "
                                "Please use the 'authorization' parameter instead."
                            )

        guardrail_ids = extract_guardrail_ids(guardrails) if guardrails else []

        # Validate that Safety API is available if guardrails are requested
        if guardrail_ids and self.safety_api is None:
            raise ValueError(
                "Cannot process guardrails: Safety API is not configured.\n\n"
                "To use guardrails, ensure the Safety API is configured in your stack, or remove "
                "the 'guardrails' parameter from your request."
            )

        if conversation is not None:
            if previous_response_id is not None:
                raise ValueError(
                    "Mutually exclusive parameters: 'previous_response_id' and 'conversation'. Ensure you are only providing one of these parameters."
                )

            if not conversation.startswith("conv_"):
                raise InvalidConversationIdError(conversation)

        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError(f"Invalid {max_tool_calls=}; should be >= 1")

        stream_gen = self._create_streaming_response(
            input=input,
            conversation=conversation,
            model=model,
            prompt=prompt,
            instructions=instructions,
            previous_response_id=previous_response_id,
            store=store,
            temperature=temperature,
            text=text,
            tools=tools,
            tool_choice=tool_choice,
            max_infer_iters=max_infer_iters,
            guardrail_ids=guardrail_ids,
            parallel_tool_calls=parallel_tool_calls,
            max_tool_calls=max_tool_calls,
            metadata=metadata,
            include=include,
        )

        if stream:
            return stream_gen
        else:
            final_response = None
            final_event_type = None
            failed_response = None

            async for stream_chunk in stream_gen:
                match stream_chunk.type:
                    case "response.completed" | "response.incomplete":
                        if final_response is not None:
                            raise ValueError(
                                "The response stream produced multiple terminal responses! "
                                f"Earlier response from {final_event_type}"
                            )
                        final_response = stream_chunk.response
                        final_event_type = stream_chunk.type
                    case "response.failed":
                        failed_response = stream_chunk.response
                    case _:
                        pass  # Other event types don't have .response

            if failed_response is not None:
                error_message = (
                    failed_response.error.message
                    if failed_response and failed_response.error
                    else "Response stream failed without error details"
                )
                raise RuntimeError(f"OpenAI response failed: {error_message}")

            if final_response is None:
                raise ValueError("The response stream never reached a terminal state")
            return final_response

    async def _create_streaming_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        prompt: OpenAIResponsePrompt | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        max_infer_iters: int | None = 10,
        guardrail_ids: list[str] | None = None,
        parallel_tool_calls: bool | None = True,
        max_tool_calls: int | None = None,
        metadata: dict[str, str] | None = None,
        include: list[ResponseItemInclude] | None = None,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # These should never be None when called from create_openai_response (which sets defaults)
        # but we assert here to help mypy understand the types
        assert text is not None, "text must not be None"
        assert max_infer_iters is not None, "max_infer_iters must not be None"

        # Input preprocessing
        all_input, messages, tool_context = await self._process_input_with_previous_response(
            input, tools, previous_response_id, conversation
        )

        if instructions:
            messages.insert(0, OpenAISystemMessageParam(content=instructions))

        # Prepend reusable prompt (if provided)
        await self._prepend_prompt(messages, prompt)

        # Structured outputs
        response_format = await convert_response_text_to_chat_response_format(text)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            response_format=response_format,
            tool_context=tool_context,
            inputs=all_input,
        )

        # Create orchestrator and delegate streaming logic
        response_id = f"resp_{uuid.uuid4()}"
        created_at = int(time.time())

        # Create a per-request MCP session manager for session reuse (fix for #4452)
        # This avoids redundant tools/list calls when making multiple MCP tool invocations
        mcp_session_manager = MCPSessionManager()

        # Create a per-request ToolExecutor with the session manager
        request_tool_executor = ToolExecutor(
            tool_groups_api=self.tool_groups_api,
            tool_runtime_api=self.tool_runtime_api,
            vector_io_api=self.vector_io_api,
            vector_stores_config=self.tool_executor.vector_stores_config,
            mcp_session_manager=mcp_session_manager,
        )

        orchestrator = StreamingResponseOrchestrator(
            inference_api=self.inference_api,
            ctx=ctx,
            response_id=response_id,
            created_at=created_at,
            prompt=prompt,
            text=text,
            max_infer_iters=max_infer_iters,
            parallel_tool_calls=parallel_tool_calls,
            tool_executor=request_tool_executor,
            safety_api=self.safety_api,
            guardrail_ids=guardrail_ids,
            instructions=instructions,
            max_tool_calls=max_tool_calls,
            metadata=metadata,
            include=include,
        )

        # Stream the response
        final_response = None
        failed_response = None

        # Type as ConversationItem to avoid list invariance issues
        output_items: list[ConversationItem] = []

        # Prepare input items for storage once (used by all persistence calls)
        input_items_for_storage = self._prepare_input_items_for_storage(all_input)

        try:
            async for stream_chunk in orchestrator.create_response():
                match stream_chunk.type:
                    case "response.completed" | "response.incomplete":
                        final_response = stream_chunk.response
                    case "response.failed":
                        failed_response = stream_chunk.response
                    case "response.output_item.done":
                        item = stream_chunk.item
                        output_items.append(item)
                    case _:
                        pass  # Other event types

                # Incremental persistence: persist on significant state changes
                # This enables clients to poll GET /v1/responses/{response_id} during streaming
                if store:
                    await self._persist_streaming_state(
                        stream_chunk=stream_chunk,
                        orchestrator=orchestrator,
                        input_items=input_items_for_storage,
                        output_items=output_items,
                    )

                # Store and sync before yielding terminal events
                # This ensures the storage/syncing happens even if the consumer breaks after receiving the event
                if (
                    stream_chunk.type in {"response.completed", "response.incomplete"}
                    and final_response
                    and failed_response is None
                ):
                    if conversation:
                        messages_to_store = list(
                            filter(lambda x: not isinstance(x, OpenAISystemMessageParam), orchestrator.final_messages)
                        )
                        await self._sync_response_to_conversation(conversation, input, output_items)
                        await self.responses_store.store_conversation_messages(conversation, messages_to_store)

                yield stream_chunk
        finally:
            # Clean up MCP sessions at the end of the request (fix for #4452)
            # Use shield() to prevent cancellation from interrupting cleanup and leaking resources
            # Wrap in try/except as cleanup errors should not mask the original response
            try:
                await asyncio.shield(mcp_session_manager.close_all())
            except BaseException as e:
                # Debug level - cleanup errors are expected in streaming scenarios where
                # anyio cancel scopes may be in a different task context
                logger.debug(f"Error during MCP session cleanup: {e}")

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        return await self.responses_store.delete_response_object(response_id)

    async def _sync_response_to_conversation(
        self, conversation_id: str, input: str | list[OpenAIResponseInput] | None, output_items: list[ConversationItem]
    ) -> None:
        """Sync content and response messages to the conversation."""
        # Type as ConversationItem union to avoid list invariance issues
        conversation_items: list[ConversationItem] = []

        if isinstance(input, str):
            conversation_items.append(
                OpenAIResponseMessage(role="user", content=[OpenAIResponseInputMessageContentText(text=input)])
            )
        elif isinstance(input, list):
            conversation_items.extend(input)

        conversation_items.extend(output_items)

        adapter = TypeAdapter(list[ConversationItem])
        validated_items = adapter.validate_python(conversation_items)
        await self.conversations_api.add_items(conversation_id, validated_items)
