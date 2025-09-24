# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal, Protocol, runtime_checkable

from openai import NOT_GIVEN
from openai._types import NotGiven
from openai.types.responses.response_includable import ResponseIncludable
from pydantic import BaseModel, Field

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageWebSearchToolCall,
)
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
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
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools,
    Field(discriminator="type"),
]
register_schema(ConversationItem, name="ConversationItem")

# from openai.types.response import *
# from openai.types.responses.response_item import *
# f = [
# ResponseFunctionToolCallItem,
# ResponseFunctionToolCallOutputItem,
# ResponseFileSearchToolCall,
# ResponseFunctionWebSearch,
# ImageGenerationCall,
# ResponseComputerToolCall,
# ResponseComputerToolCallOutputItem,
# ResponseReasoningItem,
# ResponseCodeInterpreterToolCall,
# LocalShellCall,
# LocalShellCallOutput,
# McpListTools,
# McpApprovalRequest,
# McpApprovalResponse,
# McpCall,
# ResponseCustomToolCall,
# ResponseCustomToolCallOutput
# ]


@json_schema_type
class ConversationCreateRequest(BaseModel):
    """Request body for creating a conversation."""

    items: list[ConversationItem] | None = Field(
        default=[],
        description="Initial items to include in the conversation context. You may add up to 20 items at a time.",
        max_length=20,
    )
    metadata: Metadata | None = Field(
        default={},
        description="Set of 16 key-value pairs that can be attached to an object. Useful for storing additional information",
        max_length=16,
    )


@json_schema_type
class ConversationUpdateRequest(BaseModel):
    """Request body for updating a conversation."""

    metadata: Metadata = Field(
        ...,
        description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.",
    )


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
@trace_protocol
class Conversations(Protocol):
    """Protocol for conversation management operations."""

    @webmethod(route="/v1/conversations", method="POST")
    async def create_conversation(self, request: ConversationCreateRequest) -> Conversation:
        """Create a conversation.

        :param request: The conversation creation request
        :return: The created conversation object
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}", method="GET")
    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation with the given ID.

        :param conversation_id: The conversation identifier
        :return: The conversation object
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}", method="PATCH")
    async def update_conversation(self, conversation_id: str, request: ConversationUpdateRequest) -> Conversation:
        """Update a conversation's metadata with the given ID.

        :param conversation_id: The conversation identifier
        :param request: The conversation update request
        :return: The updated conversation object
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}", method="DELETE")
    async def delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        """Delete a conversation with the given ID.

        :param conversation_id: The conversation identifier
        :return: The deleted conversation resource
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}/items", method="POST")
    async def create(self, conversation_id: str, request: ConversationItemCreateRequest) -> ConversationItemList:
        """Create items in the conversation.

        :param conversation_id: The conversation identifier
        :param request: The items creation request
        :return: List of created items
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}/items/{item_id}", method="GET")
    async def retrieve(self, conversation_id: str, item_id: str) -> ConversationItem:
        """Retrieve a conversation item.

        :param conversation_id: The conversation identifier
        :param item_id: The item identifier
        :return: The conversation item
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}/items", method="GET")
    async def list(
        self,
        conversation_id: str,
        after: str | NotGiven = NOT_GIVEN,
        include: list[ResponseIncludable] | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
        order: Literal["asc", "desc"] | NotGiven = NOT_GIVEN,
    ) -> ConversationItemList:
        """List items in the conversation.

        :param conversation_id: The conversation identifier
        :param after: An item ID to list items after, used in pagination
        :param include: Specify additional output data to include in the response
        :param limit: A limit on the number of objects to be returned (1-100, default 20)
        :param order: The order to return items in (asc or desc, default desc)
        :return: List of conversation items
        """
        ...

    @webmethod(route="/v1/conversations/{conversation_id}/items/{item_id}", method="DELETE")
    async def delete(self, conversation_id: str, item_id: str) -> ConversationItemDeletedResource:
        """Delete a conversation item.

        :param conversation_id: The conversation identifier
        :param item_id: The item identifier
        :return: The deleted item resource
        """
        ...
