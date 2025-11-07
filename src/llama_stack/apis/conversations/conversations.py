# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageWebSearchToolCall,
)
from llama_stack.apis.common.tracing import telemetry_traceable
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod

Metadata = dict[str, str]


@json_schema_type
class Conversation(BaseModel):
    """OpenAI-compatible conversation object."""

    id: str = Field(..., description="The unique ID of the conversation.")
    object: Literal["conversation"] = Field(
        default="conversation", description="The object type, which is always conversation."
    )
    created_at: int = Field(
        ..., description="The time at which the conversation was created, measured in seconds since the Unix epoch."
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.",
    )
    items: list[dict] | None = Field(
        default=None,
        description="Initial items to include in the conversation context. You may add up to 20 items at a time.",
    )


@json_schema_type
class ConversationMessage(BaseModel):
    """OpenAI-compatible message item for conversations."""

    id: str = Field(..., description="unique identifier for this message")
    content: list[dict] = Field(..., description="message content")
    role: str = Field(..., description="message role")
    status: str = Field(..., description="message status")
    type: Literal["message"] = "message"
    object: Literal["message"] = "message"


ConversationItem = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseInputFunctionToolCallOutput
    | OpenAIResponseMCPApprovalRequest
    | OpenAIResponseMCPApprovalResponse
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools,
    Field(discriminator="type"),
]
register_schema(ConversationItem, name="ConversationItem")

# Using OpenAI types directly caused issues but some notes for reference:
# Note that ConversationItem is a Annotated Union of the types below:
# from openai.types.responses import *
# from openai.types.responses.response_item import *
# from openai.types.conversations import ConversationItem
# f = [
#     ResponseFunctionToolCallItem,
#     ResponseFunctionToolCallOutputItem,
#     ResponseFileSearchToolCall,
#     ResponseFunctionWebSearch,
#     ImageGenerationCall,
#     ResponseComputerToolCall,
#     ResponseComputerToolCallOutputItem,
#     ResponseReasoningItem,
#     ResponseCodeInterpreterToolCall,
#     LocalShellCall,
#     LocalShellCallOutput,
#     McpListTools,
#     McpApprovalRequest,
#     McpApprovalResponse,
#     McpCall,
#     ResponseCustomToolCall,
#     ResponseCustomToolCallOutput
# ]


@json_schema_type
class ConversationDeletedResource(BaseModel):
    """Response for deleted conversation."""

    id: str = Field(..., description="The deleted conversation identifier")
    object: str = Field(default="conversation.deleted", description="Object type")
    deleted: bool = Field(default=True, description="Whether the object was deleted")


@json_schema_type
class ConversationItemCreateRequest(BaseModel):
    """Request body for creating conversation items."""

    items: list[ConversationItem] = Field(
        ...,
        description="Items to include in the conversation context. You may add up to 20 items at a time.",
        max_length=20,
    )


class ConversationItemInclude(StrEnum):
    """
    Specify additional output data to include in the model response.
    """

    web_search_call_action_sources = "web_search_call.action.sources"
    code_interpreter_call_outputs = "code_interpreter_call.outputs"
    computer_call_output_output_image_url = "computer_call_output.output.image_url"
    file_search_call_results = "file_search_call.results"
    message_input_image_image_url = "message.input_image.image_url"
    message_output_text_logprobs = "message.output_text.logprobs"
    reasoning_encrypted_content = "reasoning.encrypted_content"


@json_schema_type
class ConversationItemList(BaseModel):
    """List of conversation items with pagination."""

    object: str = Field(default="list", description="Object type")
    data: list[ConversationItem] = Field(..., description="List of conversation items")
    first_id: str | None = Field(default=None, description="The ID of the first item in the list")
    last_id: str | None = Field(default=None, description="The ID of the last item in the list")
    has_more: bool = Field(default=False, description="Whether there are more items available")


@json_schema_type
class ConversationItemDeletedResource(BaseModel):
    """Response for deleted conversation item."""

    id: str = Field(..., description="The deleted item identifier")
    object: str = Field(default="conversation.item.deleted", description="Object type")
    deleted: bool = Field(default=True, description="Whether the object was deleted")


@runtime_checkable
@telemetry_traceable
class Conversations(Protocol):
    """Conversations

    Protocol for conversation management operations."""

    @webmethod(route="/conversations", method="POST", level=LLAMA_STACK_API_V1)
    async def create_conversation(
        self, items: list[ConversationItem] | None = None, metadata: Metadata | None = None
    ) -> Conversation:
        """Create a conversation.

        Create a conversation.

        :param items: Initial items to include in the conversation context.
        :param metadata: Set of key-value pairs that can be attached to an object.
        :returns: The created conversation object.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve a conversation.

        Get a conversation with the given ID.

        :param conversation_id: The conversation identifier.
        :returns: The conversation object.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}", method="POST", level=LLAMA_STACK_API_V1)
    async def update_conversation(self, conversation_id: str, metadata: Metadata) -> Conversation:
        """Update a conversation.

        Update a conversation's metadata with the given ID.

        :param conversation_id: The conversation identifier.
        :param metadata: Set of key-value pairs that can be attached to an object.
        :returns: The updated conversation object.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}", method="DELETE", level=LLAMA_STACK_API_V1)
    async def openai_delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        """Delete a conversation.

        Delete a conversation with the given ID.

        :param conversation_id: The conversation identifier.
        :returns: The deleted conversation resource.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}/items", method="POST", level=LLAMA_STACK_API_V1)
    async def add_items(self, conversation_id: str, items: list[ConversationItem]) -> ConversationItemList:
        """Create items.

        Create items in the conversation.

        :param conversation_id: The conversation identifier.
        :param items: Items to include in the conversation context.
        :returns: List of created items.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}/items/{item_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def retrieve(self, conversation_id: str, item_id: str) -> ConversationItem:
        """Retrieve an item.

        Retrieve a conversation item.

        :param conversation_id: The conversation identifier.
        :param item_id: The item identifier.
        :returns: The conversation item.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}/items", method="GET", level=LLAMA_STACK_API_V1)
    async def list_items(
        self,
        conversation_id: str,
        after: str | None = None,
        include: list[ConversationItemInclude] | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
    ) -> ConversationItemList:
        """List items.

        List items in the conversation.

        :param conversation_id: The conversation identifier.
        :param after: An item ID to list items after, used in pagination.
        :param include: Specify additional output data to include in the response.
        :param limit: A limit on the number of objects to be returned (1-100, default 20).
        :param order: The order to return items in (asc or desc, default desc).
        :returns: List of conversation items.
        """
        ...

    @webmethod(route="/conversations/{conversation_id}/items/{item_id}", method="DELETE", level=LLAMA_STACK_API_V1)
    async def openai_delete_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> ConversationItemDeletedResource:
        """Delete an item.

        Delete a conversation item.

        :param conversation_id: The conversation identifier.
        :param item_id: The item identifier.
        :returns: The deleted item resource.
        """
        ...
