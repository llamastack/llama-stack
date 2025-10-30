# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import re
import uuid
from collections.abc import Sequence

from llama_stack.apis.agents.agents import ResponseGuardrailSpec
from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseInput,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContent,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseText,
)
from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIDeveloperMessageParam,
    OpenAIImageURL,
    OpenAIJSONSchema,
    OpenAIMessageParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIResponseFormatParam,
    OpenAIResponseFormatText,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.safety import Safety


async def convert_chat_choice_to_response_message(
    choice: OpenAIChoice,
    citation_files: dict[str, str] | None = None,
    *,
    message_id: str | None = None,
) -> OpenAIResponseMessage:
    """Convert an OpenAI Chat Completion choice into an OpenAI Response output message."""
    output_content = ""
    if isinstance(choice.message.content, str):
        output_content = choice.message.content
    elif isinstance(choice.message.content, OpenAIChatCompletionContentPartTextParam):
        output_content = choice.message.content.text
    else:
        raise ValueError(
            f"Llama Stack OpenAI Responses does not yet support output content type: {type(choice.message.content)}"
        )

    annotations, clean_text = _extract_citations_from_text(output_content, citation_files or {})

    return OpenAIResponseMessage(
        id=message_id or f"msg_{uuid.uuid4()}",
        content=[OpenAIResponseOutputMessageContentOutputText(text=clean_text, annotations=list(annotations))],
        status="completed",
        role="assistant",
    )


async def convert_response_content_to_chat_content(
    content: str | Sequence[OpenAIResponseInputMessageContent | OpenAIResponseOutputMessageContent],
) -> str | list[OpenAIChatCompletionContentPartParam]:
    """
    Convert the content parts from an OpenAI Response API request into OpenAI Chat Completion content parts.

    The content schemas of each API look similar, but are not exactly the same.
    """
    if isinstance(content, str):
        return content

    # Type with union to avoid list invariance issues
    converted_parts: list[OpenAIChatCompletionContentPartParam] = []
    for content_part in content:
        if isinstance(content_part, OpenAIResponseInputMessageContentText):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part.text))
        elif isinstance(content_part, OpenAIResponseOutputMessageContentOutputText):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part.text))
        elif isinstance(content_part, OpenAIResponseInputMessageContentImage):
            if content_part.image_url:
                image_url = OpenAIImageURL(url=content_part.image_url, detail=content_part.detail)
                converted_parts.append(OpenAIChatCompletionContentPartImageParam(image_url=image_url))
        elif isinstance(content_part, str):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part))
        else:
            raise ValueError(
                f"Llama Stack OpenAI Responses does not yet support content type '{type(content_part)}' in this context"
            )
    return converted_parts


async def convert_response_input_to_chat_messages(
    input: str | list[OpenAIResponseInput],
    previous_messages: list[OpenAIMessageParam] | None = None,
) -> list[OpenAIMessageParam]:
    """
    Convert the input from an OpenAI Response API request into OpenAI Chat Completion messages.

    :param input: The input to convert
    :param previous_messages: Optional previous messages to check for function_call references
    """
    messages: list[OpenAIMessageParam] = []
    if isinstance(input, list):
        # extract all OpenAIResponseInputFunctionToolCallOutput items
        # so their corresponding OpenAIToolMessageParam instances can
        # be added immediately following the corresponding
        # OpenAIAssistantMessageParam
        tool_call_results = {}
        for input_item in input:
            if isinstance(input_item, OpenAIResponseInputFunctionToolCallOutput):
                tool_call_results[input_item.call_id] = OpenAIToolMessageParam(
                    content=input_item.output,
                    tool_call_id=input_item.call_id,
                )

        for input_item in input:
            if isinstance(input_item, OpenAIResponseInputFunctionToolCallOutput):
                # skip as these have been extracted and inserted in order
                pass
            elif isinstance(input_item, OpenAIResponseOutputMessageFunctionToolCall):
                tool_call = OpenAIChatCompletionToolCall(
                    index=0,
                    id=input_item.call_id,
                    function=OpenAIChatCompletionToolCallFunction(
                        name=input_item.name,
                        arguments=input_item.arguments,
                    ),
                )
                messages.append(OpenAIAssistantMessageParam(tool_calls=[tool_call]))
                if input_item.call_id in tool_call_results:
                    messages.append(tool_call_results[input_item.call_id])
                    del tool_call_results[input_item.call_id]
            elif isinstance(input_item, OpenAIResponseOutputMessageMCPCall):
                tool_call = OpenAIChatCompletionToolCall(
                    index=0,
                    id=input_item.id,
                    function=OpenAIChatCompletionToolCallFunction(
                        name=input_item.name,
                        arguments=input_item.arguments,
                    ),
                )
                messages.append(OpenAIAssistantMessageParam(tool_calls=[tool_call]))
                # Output can be None, use empty string as fallback
                output_content = input_item.output if input_item.output is not None else ""
                messages.append(
                    OpenAIToolMessageParam(
                        content=output_content,
                        tool_call_id=input_item.id,
                    )
                )
            elif isinstance(input_item, OpenAIResponseOutputMessageMCPListTools):
                # the tool list will be handled separately
                pass
            elif isinstance(input_item, OpenAIResponseMCPApprovalRequest) or isinstance(
                input_item, OpenAIResponseMCPApprovalResponse
            ):
                # these are handled by the responses impl itself and not pass through to chat completions
                pass
            elif isinstance(input_item, OpenAIResponseMessage):
                # Narrow type to OpenAIResponseMessage which has content and role attributes
                content = await convert_response_content_to_chat_content(input_item.content)
                message_type = await get_message_type_by_role(input_item.role)
                if message_type is None:
                    raise ValueError(
                        f"Llama Stack OpenAI Responses does not yet support message role '{input_item.role}' in this context"
                    )
                # Skip user messages that duplicate the last user message in previous_messages
                # This handles cases where input includes context for function_call_outputs
                if previous_messages and input_item.role == "user":
                    last_user_msg = None
                    for msg in reversed(previous_messages):
                        if isinstance(msg, OpenAIUserMessageParam):
                            last_user_msg = msg
                            break
                    if last_user_msg:
                        last_user_content = getattr(last_user_msg, "content", None)
                        if last_user_content == content:
                            continue  # Skip duplicate user message
                # Dynamic message type call - different message types have different content expectations
                messages.append(message_type(content=content))  # type: ignore[call-arg,arg-type]
        if len(tool_call_results):
            # Check if unpaired function_call_outputs reference function_calls from previous messages
            if previous_messages:
                previous_call_ids = _extract_tool_call_ids(previous_messages)
                for call_id in list(tool_call_results.keys()):
                    if call_id in previous_call_ids:
                        # Valid: this output references a call from previous messages
                        # Add the tool message
                        messages.append(tool_call_results[call_id])
                        del tool_call_results[call_id]

            # If still have unpaired outputs, error
            if len(tool_call_results):
                raise ValueError(
                    f"Received function_call_output(s) with call_id(s) {tool_call_results.keys()}, but no corresponding function_call"
                )
    else:
        messages.append(OpenAIUserMessageParam(content=input))
    return messages


def _extract_tool_call_ids(messages: list[OpenAIMessageParam]) -> set[str]:
    """Extract all tool_call IDs from messages."""
    call_ids = set()
    for msg in messages:
        if isinstance(msg, OpenAIAssistantMessageParam):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    # tool_call is a Pydantic model, use attribute access
                    call_ids.add(tool_call.id)
    return call_ids


async def convert_response_text_to_chat_response_format(
    text: OpenAIResponseText,
) -> OpenAIResponseFormatParam:
    """
    Convert an OpenAI Response text parameter into an OpenAI Chat Completion response format.
    """
    if not text.format or text.format["type"] == "text":
        return OpenAIResponseFormatText(type="text")
    if text.format["type"] == "json_object":
        return OpenAIResponseFormatJSONObject()
    if text.format["type"] == "json_schema":
        # Assert name exists for json_schema format
        assert text.format.get("name"), "json_schema format requires a name"
        schema_name: str = text.format["name"]  # type: ignore[assignment]
        return OpenAIResponseFormatJSONSchema(
            json_schema=OpenAIJSONSchema(name=schema_name, schema=text.format["schema"])
        )
    raise ValueError(f"Unsupported text format: {text.format}")


async def get_message_type_by_role(role: str) -> type[OpenAIMessageParam] | None:
    """Get the appropriate OpenAI message parameter type for a given role."""
    role_to_type = {
        "user": OpenAIUserMessageParam,
        "system": OpenAISystemMessageParam,
        "assistant": OpenAIAssistantMessageParam,
        "developer": OpenAIDeveloperMessageParam,
    }
    return role_to_type.get(role)  # type: ignore[return-value]  # Pydantic models use ModelMetaclass


def _extract_citations_from_text(
    text: str, citation_files: dict[str, str]
) -> tuple[list[OpenAIResponseAnnotationFileCitation], str]:
    """Extract citation markers from text and create annotations

    Args:
        text: The text containing citation markers like [file-Cn3MSNn72ENTiiq11Qda4A]
        citation_files: Dictionary mapping file_id to filename

    Returns:
        Tuple of (annotations_list, clean_text_without_markers)
    """
    file_id_regex = re.compile(r"<\|(?P<file_id>file-[A-Za-z0-9_-]+)\|>")

    annotations = []
    parts = []
    total_len = 0
    last_end = 0

    for m in file_id_regex.finditer(text):
        # segment before the marker
        prefix = text[last_end : m.start()]

        # drop one space if it exists (since marker is at sentence end)
        if prefix.endswith(" "):
            prefix = prefix[:-1]

        parts.append(prefix)
        total_len += len(prefix)

        fid = m.group(1)
        if fid in citation_files:
            annotations.append(
                OpenAIResponseAnnotationFileCitation(
                    file_id=fid,
                    filename=citation_files[fid],
                    index=total_len,  # index points to punctuation
                )
            )

        last_end = m.end()

    parts.append(text[last_end:])
    cleaned_text = "".join(parts)
    return annotations, cleaned_text


def is_function_tool_call(
    tool_call: OpenAIChatCompletionToolCall,
    tools: list[OpenAIResponseInputTool],
) -> bool:
    if not tool_call.function:
        return False
    for t in tools:
        if t.type == "function" and t.name == tool_call.function.name:
            return True
    return False


async def run_guardrails(safety_api: Safety, messages: str, guardrail_ids: list[str]) -> str | None:
    """Run guardrails against messages and return violation message if blocked."""
    if not messages:
        return None

    # Look up shields to get their provider_resource_id (actual model ID)
    model_ids = []
    # TODO: list_shields not in Safety interface but available at runtime via API routing
    shields_list = await safety_api.routing_table.list_shields()  # type: ignore[attr-defined]

    for guardrail_id in guardrail_ids:
        matching_shields = [shield for shield in shields_list.data if shield.identifier == guardrail_id]
        if matching_shields:
            model_id = matching_shields[0].provider_resource_id
            model_ids.append(model_id)
        else:
            raise ValueError(f"No shield found with identifier '{guardrail_id}'")

    guardrail_tasks = [safety_api.run_moderation(messages, model=model_id) for model_id in model_ids]
    responses = await asyncio.gather(*guardrail_tasks)

    for response in responses:
        for result in response.results:
            if result.flagged:
                message = result.user_message or "Content blocked by safety guardrails"
                flagged_categories = (
                    [cat for cat, flagged in result.categories.items() if flagged] if result.categories else []
                )
                violation_type = result.metadata.get("violation_type", []) if result.metadata else []

                if flagged_categories:
                    message += f" (flagged for: {', '.join(flagged_categories)})"
                if violation_type:
                    message += f" (violation type: {', '.join(violation_type)})"

                return message

    # No violations found
    return None


def extract_guardrail_ids(guardrails: list | None) -> list[str]:
    """Extract guardrail IDs from guardrails parameter, handling both string IDs and ResponseGuardrailSpec objects."""
    if not guardrails:
        return []

    guardrail_ids = []
    for guardrail in guardrails:
        if isinstance(guardrail, str):
            guardrail_ids.append(guardrail)
        elif isinstance(guardrail, ResponseGuardrailSpec):
            guardrail_ids.append(guardrail.type)
        else:
            raise ValueError(f"Unknown guardrail format: {guardrail}, expected str or ResponseGuardrailSpec")

    return guardrail_ids
