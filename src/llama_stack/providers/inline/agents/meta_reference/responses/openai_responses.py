# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from collections.abc import AsyncIterator

from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.agents import Order
from llama_stack.apis.agents.agents import ResponseGuardrailSpec
from llama_stack.apis.agents.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)
from llama_stack.apis.common.errors import (
    InvalidConversationIdError,
)
from llama_stack.apis.conversations import Conversations
from llama_stack.apis.conversations.conversations import ConversationItem
from llama_stack.apis.inference import (
    Inference,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.log import get_logger
from llama_stack.providers.utils.responses.responses_store import (
    ResponsesStore,
    _OpenAIResponseObjectWithInputAndMessages,
)

from .streaming import StreamingResponseOrchestrator
from .tool_executor import ToolExecutor
from .types import ChatCompletionContext, ToolContext
from .utils import (
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
        safety_api: Safety,
        conversations_api: Conversations,
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
        )

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
                new_messages = await convert_response_input_to_chat_messages(input, previous_messages=messages)
                messages.extend(new_messages)
            else:
                # Backward compatibility: reconstruct from inputs
                messages = await convert_response_input_to_chat_messages(all_input)

            tool_context.recover_tools_from_previous_response(previous_response)
        elif conversation is not None:
            conversation_items = await self.conversations_api.list_items(conversation, order="asc")

            # Use stored messages as source of truth (like previous_response.messages)
            stored_messages = await self.responses_store.get_conversation_messages(conversation)

            all_input = input
            if not conversation_items.data:
                # First turn - just convert the new input
                messages = await convert_response_input_to_chat_messages(input)
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
                new_messages = await convert_response_input_to_chat_messages(all_input, previous_messages=messages)
                messages.extend(new_messages)
        else:
            all_input = input
            messages = await convert_response_input_to_chat_messages(all_input)

        return all_input, messages, tool_context

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
        include: list[str] | None = None,
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
        guardrails: list[str | ResponseGuardrailSpec] | None = None,
        max_tool_calls: int | None = None,
    ):
        stream = bool(stream)
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")) if text is None else text

        guardrail_ids = extract_guardrail_ids(guardrails) if guardrails else []

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
            instructions=instructions,
            previous_response_id=previous_response_id,
            store=store,
            temperature=temperature,
            text=text,
            tools=tools,
            max_infer_iters=max_infer_iters,
            guardrail_ids=guardrail_ids,
            max_tool_calls=max_tool_calls,
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
        store: bool | None = True,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        max_infer_iters: int | None = 10,
        guardrail_ids: list[str] | None = None,
        max_tool_calls: int | None = None,
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

        # Structured outputs
        response_format = await convert_response_text_to_chat_response_format(text)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            temperature=temperature,
            response_format=response_format,
            tool_context=tool_context,
            inputs=all_input,
        )

        # Create orchestrator and delegate streaming logic
        response_id = f"resp_{uuid.uuid4()}"
        created_at = int(time.time())

        orchestrator = StreamingResponseOrchestrator(
            inference_api=self.inference_api,
            ctx=ctx,
            response_id=response_id,
            created_at=created_at,
            text=text,
            max_infer_iters=max_infer_iters,
            tool_executor=self.tool_executor,
            safety_api=self.safety_api,
            guardrail_ids=guardrail_ids,
            instructions=instructions,
            max_tool_calls=max_tool_calls,
        )

        # Stream the response
        final_response = None
        failed_response = None

        # Type as ConversationItem to avoid list invariance issues
        output_items: list[ConversationItem] = []
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

            # Store and sync before yielding terminal events
            # This ensures the storage/syncing happens even if the consumer breaks after receiving the event
            if (
                stream_chunk.type in {"response.completed", "response.incomplete"}
                and final_response
                and failed_response is None
            ):
                messages_to_store = list(
                    filter(lambda x: not isinstance(x, OpenAISystemMessageParam), orchestrator.final_messages)
                )
                if store:
                    # TODO: we really should work off of output_items instead of "final_messages"
                    await self._store_response(
                        response=final_response,
                        input=all_input,
                        messages=messages_to_store,
                    )

                if conversation:
                    await self._sync_response_to_conversation(conversation, input, output_items)
                    await self.responses_store.store_conversation_messages(conversation, messages_to_store)

            yield stream_chunk

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
