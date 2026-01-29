# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Agents API requests and responses.

This module defines the request and response models for the Agents API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from llama_stack_api.common.responses import Order
from llama_stack_api.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponsePrompt,
    OpenAIResponseReasoning,
    OpenAIResponseText,
)


class ResponseItemInclude(StrEnum):
    """Specify additional output data to include in the model response."""

    web_search_call_action_sources = "web_search_call.action.sources"
    code_interpreter_call_outputs = "code_interpreter_call.outputs"
    computer_call_output_output_image_url = "computer_call_output.output.image_url"
    file_search_call_results = "file_search_call.results"
    message_input_image_image_url = "message.input_image.image_url"
    message_output_text_logprobs = "message.output_text.logprobs"
    reasoning_encrypted_content = "reasoning.encrypted_content"


class ResponseGuardrailSpec(BaseModel):
    """Specification for a guardrail to apply during response generation."""

    model_config = ConfigDict(extra="forbid")

    type: str
    # TODO: more fields to be added for guardrail configuration


ResponseGuardrail = str | ResponseGuardrailSpec


class CreateResponseRequest(BaseModel):
    """Request model for creating a response."""

    model_config = ConfigDict(extra="forbid")

    input: str | list[OpenAIResponseInput] = Field(..., description="Input message(s) to create the response.")
    model: str = Field(..., description="The underlying LLM used for completions.")
    prompt: OpenAIResponsePrompt | None = Field(
        default=None, description="Prompt object with ID, version, and variables."
    )
    instructions: str | None = Field(default=None, description="Instructions to guide the model's behavior.")
    parallel_tool_calls: bool | None = Field(
        default=True,
        description="Whether to enable parallel tool calls.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Optional ID of a previous response to continue from.",
    )
    conversation: str | None = Field(
        default=None,
        description="Optional ID of a conversation to add the response to.",
    )
    store: bool | None = Field(
        default=True,
        description="Whether to store the response in the database.",
    )
    stream: bool | None = Field(
        default=False,
        description="Whether to stream the response.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    text: OpenAIResponseText | None = Field(
        default=None,
        description="Configuration for text response generation.",
    )
    tool_choice: OpenAIResponseInputToolChoice | None = Field(
        default=None,
        description="How the model should select which tool to call (if any).",
    )
    tools: list[OpenAIResponseInputTool] | None = Field(
        default=None,
        description="List of tools available to the model.",
    )
    include: list[ResponseItemInclude] | None = Field(
        default=None,
        description="Additional fields to include in the response.",
    )
    max_infer_iters: int | None = Field(
        default=10,
        ge=1,
        description="Maximum number of inference iterations.",
    )
    guardrails: list[ResponseGuardrail] | None = Field(
        default=None,
        description="List of guardrails to apply during response generation.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        description="Max number of total calls to built-in tools that can be processed in a response.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Upper bound for the number of tokens that can be generated for a response.",
    )
    reasoning: OpenAIResponseReasoning | None = Field(
        default=None,
        description="Configuration for reasoning effort in responses.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Dictionary of metadata key-value pairs to attach to the response.",
    )


class RetrieveResponseRequest(BaseModel):
    """Request model for retrieving a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the OpenAI response to retrieve.")


class ListResponsesRequest(BaseModel):
    """Request model for listing responses."""

    model_config = ConfigDict(extra="forbid")

    after: str | None = Field(default=None, description="The ID of the last response to return.")
    limit: int | None = Field(default=50, ge=1, le=100, description="The number of responses to return.")
    model: str | None = Field(default=None, description="The model to filter responses by.")
    order: Order | None = Field(
        default=Order.desc,
        description="The order to sort responses by when sorted by created_at ('asc' or 'desc').",
    )


class ListResponseInputItemsRequest(BaseModel):
    """Request model for listing input items of a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the response to retrieve input items for.")
    after: str | None = Field(default=None, description="An item ID to list items after, used for pagination.")
    before: str | None = Field(default=None, description="An item ID to list items before, used for pagination.")
    include: list[ResponseItemInclude] | None = Field(
        default=None, description="Additional fields to include in the response."
    )
    limit: int | None = Field(
        default=20,
        ge=1,
        le=100,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
    )
    order: Order | None = Field(default=Order.desc, description="The order to return the input items in.")


class DeleteResponseRequest(BaseModel):
    """Request model for deleting a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the OpenAI response to delete.")
