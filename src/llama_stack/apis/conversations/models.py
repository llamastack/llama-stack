# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Annotated, Literal

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
from llama_stack.schema_utils import json_schema_type, register_schema

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
