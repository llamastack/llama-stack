# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.common.responses import Order
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import ExtraBodyField, json_schema_type, webmethod

from .openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseText,
)


@json_schema_type
class ResponseGuardrailSpec(BaseModel):
    """Specification for a guardrail to apply during response generation.

    :param type: The type/identifier of the guardrail.
    """

    type: str
    # TODO: more fields to be added for guardrail configuration


ResponseGuardrail = str | ResponseGuardrailSpec


@runtime_checkable
class Agents(Protocol):
    """Agents

    APIs for creating and interacting with agentic systems."""

    # We situate the OpenAI Responses API in the Agents API just like we did things
    # for Inference. The Responses API, in its intent, serves the same purpose as
    # the Agents API above -- it is essentially a lightweight "agentic loop" with
    # integrated tool calling.
    #
    # Both of these APIs are inherently stateful.

    @webmethod(route="/responses/{response_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        """Get a model response.

        :param response_id: The ID of the OpenAI response to retrieve.
        :returns: An OpenAIResponseObject.
        """
        ...

    @webmethod(route="/responses", method="POST", level=LLAMA_STACK_API_V1)
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
        max_infer_iters: int | None = 10,  # this is an extension to the OpenAI API
        guardrails: Annotated[
            list[ResponseGuardrail] | None,
            ExtraBodyField(
                "List of guardrails to apply during response generation. Guardrails provide safety and content moderation."
            ),
        ] = None,
        max_tool_calls: int | None = None,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create a model response.

        :param input: Input message(s) to create the response.
        :param model: The underlying LLM used for completions.
        :param prompt: (Optional) Prompt object with ID, version, and variables.
        :param previous_response_id: (Optional) if specified, the new response will be a continuation of the previous response. This can be used to easily fork-off new responses from existing responses.
        :param conversation: (Optional) The ID of a conversation to add the response to. Must begin with 'conv_'. Input and output messages will be automatically added to the conversation.
        :param include: (Optional) Additional fields to include in the response.
        :param guardrails: (Optional) List of guardrails to apply during response generation. Can be guardrail IDs (strings) or guardrail specifications.
        :param max_tool_calls: (Optional) Max number of total calls to built-in tools that can be processed in a response.
        :returns: An OpenAIResponseObject.
        """
        ...

    @webmethod(route="/responses", method="GET", level=LLAMA_STACK_API_V1)
    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """List all responses.

        :param after: The ID of the last response to return.
        :param limit: The number of responses to return.
        :param model: The model to filter responses by.
        :param order: The order to sort responses by when sorted by created_at ('asc' or 'desc').
        :returns: A ListOpenAIResponseObject.
        """
        ...

    @webmethod(route="/responses/{response_id}/input_items", method="GET", level=LLAMA_STACK_API_V1)
    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
        :param order: The order to return the input items in. Default is desc.
        :returns: An ListOpenAIResponseInputItem.
        """
        ...

    @webmethod(route="/responses/{response_id}", method="DELETE", level=LLAMA_STACK_API_V1)
    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        """Delete a response.

        :param response_id: The ID of the OpenAI response to delete.
        :returns: An OpenAIDeleteResponseObject
        """
        ...
