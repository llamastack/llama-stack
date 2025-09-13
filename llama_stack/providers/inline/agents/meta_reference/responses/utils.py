# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
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


async def convert_chat_choice_to_response_message(choice: OpenAIChoice) -> OpenAIResponseMessage:
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

    return OpenAIResponseMessage(
        id=f"msg_{uuid.uuid4()}",
        content=[OpenAIResponseOutputMessageContentOutputText(text=output_content)],
        status="completed",
        role="assistant",
    )


async def convert_response_content_to_chat_content(
    content: (str | list[OpenAIResponseInputMessageContent] | list[OpenAIResponseOutputMessageContent]),
) -> str | list[OpenAIChatCompletionContentPartParam]:
    """
    Convert the content parts from an OpenAI Response API request into OpenAI Chat Completion content parts.

    The content schemas of each API look similar, but are not exactly the same.
    """
    if isinstance(content, str):
        return content

    converted_parts = []
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
) -> list[OpenAIMessageParam]:
    """
    Convert the input from an OpenAI Response API request into OpenAI Chat Completion messages.
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
                messages.append(
                    OpenAIToolMessageParam(
                        content=input_item.output,
                        tool_call_id=input_item.id,
                    )
                )
            elif isinstance(input_item, OpenAIResponseOutputMessageMCPListTools):
                # Workaround for now, to be removed when MCP list tools is supported
                tool_count = len(input_item.tools)
                content_text = f"Discovered {tool_count} tools from MCP server '{input_item.server_label}'"
                messages.append(OpenAIAssistantMessageParam(content=content_text))
            else:
                content = await convert_response_content_to_chat_content(input_item.content)
                message_type = await get_message_type_by_role(input_item.role)
                if message_type is None:
                    raise ValueError(
                        f"Llama Stack OpenAI Responses does not yet support message role '{input_item.role}' in this context"
                    )
                messages.append(message_type(content=content))
        if len(tool_call_results):
            raise ValueError(
                f"Received function_call_output(s) with call_id(s) {tool_call_results.keys()}, but no corresponding function_call"
            )
    else:
        messages.append(OpenAIUserMessageParam(content=input))
    return messages


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
        return OpenAIResponseFormatJSONSchema(
            json_schema=OpenAIJSONSchema(name=text.format["name"], schema=text.format["schema"])
        )
    raise ValueError(f"Unsupported text format: {text.format}")


async def get_message_type_by_role(role: str):
    role_to_type = {
        "user": OpenAIUserMessageParam,
        "system": OpenAISystemMessageParam,
        "assistant": OpenAIAssistantMessageParam,
        "developer": OpenAIDeveloperMessageParam,
    }
    return role_to_type.get(role)


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
