# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from llama_stack.apis.common.content_types import TextContentItem
from llama_stack.apis.inference import (
    CompletionMessage,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIDeveloperMessageParam,
    OpenAIImageURL,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
    SystemMessage,
    UserMessage,
)
from llama_stack.models.llama.datatypes import BuiltinTool, StopReason, ToolCall
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    convert_message_to_openai_dict_new,
    openai_messages_to_messages,
)


async def test_convert_message_to_openai_dict():
    message = UserMessage(content=[TextContentItem(text="Hello, world!")], role="user")
    assert await convert_message_to_openai_dict(message) == {
        "role": "user",
        "content": [{"type": "text", "text": "Hello, world!"}],
    }


# Test convert_message_to_openai_dict with a tool call
async def test_convert_message_to_openai_dict_with_tool_call():
    message = CompletionMessage(
        content="",
        tool_calls=[ToolCall(call_id="123", tool_name="test_tool", arguments='{"foo": "bar"}')],
        stop_reason=StopReason.end_of_turn,
    )

    openai_dict = await convert_message_to_openai_dict(message)

    assert openai_dict == {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"name": "test_tool", "arguments": '{"foo": "bar"}'}}
        ],
    }


async def test_convert_message_to_openai_dict_with_builtin_tool_call():
    message = CompletionMessage(
        content="",
        tool_calls=[
            ToolCall(
                call_id="123",
                tool_name=BuiltinTool.brave_search,
                arguments='{"foo": "bar"}',
            )
        ],
        stop_reason=StopReason.end_of_turn,
    )

    openai_dict = await convert_message_to_openai_dict(message)

    assert openai_dict == {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"name": "brave_search", "arguments": '{"foo": "bar"}'}}
        ],
    }


async def test_openai_messages_to_messages_with_content_str():
    openai_messages = [
        OpenAISystemMessageParam(content="system message"),
        OpenAIUserMessageParam(content="user message"),
        OpenAIAssistantMessageParam(content="assistant message"),
    ]

    llama_messages = openai_messages_to_messages(openai_messages)
    assert len(llama_messages) == 3
    assert isinstance(llama_messages[0], SystemMessage)
    assert isinstance(llama_messages[1], UserMessage)
    assert isinstance(llama_messages[2], CompletionMessage)
    assert llama_messages[0].content == "system message"
    assert llama_messages[1].content == "user message"
    assert llama_messages[2].content == "assistant message"


async def test_openai_messages_to_messages_with_content_list():
    openai_messages = [
        OpenAISystemMessageParam(content=[OpenAIChatCompletionContentPartTextParam(text="system message")]),
        OpenAIUserMessageParam(content=[OpenAIChatCompletionContentPartTextParam(text="user message")]),
        OpenAIAssistantMessageParam(content=[OpenAIChatCompletionContentPartTextParam(text="assistant message")]),
    ]

    llama_messages = openai_messages_to_messages(openai_messages)
    assert len(llama_messages) == 3
    assert isinstance(llama_messages[0], SystemMessage)
    assert isinstance(llama_messages[1], UserMessage)
    assert isinstance(llama_messages[2], CompletionMessage)
    assert llama_messages[0].content[0].text == "system message"
    assert llama_messages[1].content[0].text == "user message"
    assert llama_messages[2].content[0].text == "assistant message"


@pytest.mark.parametrize(
    "message_class,kwargs",
    [
        (OpenAISystemMessageParam, {}),
        (OpenAIAssistantMessageParam, {}),
        (OpenAIDeveloperMessageParam, {}),
        (OpenAIUserMessageParam, {}),
        (OpenAIToolMessageParam, {"tool_call_id": "call_123"}),
    ],
)
def test_message_accepts_text_string(message_class, kwargs):
    """Test that messages accept string text content."""
    msg = message_class(content="Test message", **kwargs)
    assert msg.content == "Test message"


@pytest.mark.parametrize(
    "message_class,kwargs",
    [
        (OpenAISystemMessageParam, {}),
        (OpenAIAssistantMessageParam, {}),
        (OpenAIDeveloperMessageParam, {}),
        (OpenAIUserMessageParam, {}),
        (OpenAIToolMessageParam, {"tool_call_id": "call_123"}),
    ],
)
def test_message_accepts_text_list(message_class, kwargs):
    """Test that messages accept list of text content parts."""
    content_list = [OpenAIChatCompletionContentPartTextParam(text="Test message")]
    msg = message_class(content=content_list, **kwargs)
    assert len(msg.content) == 1
    assert msg.content[0].text == "Test message"


@pytest.mark.parametrize(
    "message_class,kwargs",
    [
        (OpenAISystemMessageParam, {}),
        (OpenAIAssistantMessageParam, {}),
        (OpenAIDeveloperMessageParam, {}),
        (OpenAIToolMessageParam, {"tool_call_id": "call_123"}),
    ],
)
def test_message_rejects_images(message_class, kwargs):
    """Test that system, assistant, developer, and tool messages reject image content."""
    with pytest.raises(ValidationError):
        message_class(
            content=[
                OpenAIChatCompletionContentPartImageParam(image_url=OpenAIImageURL(url="http://example.com/image.jpg"))
            ],
            **kwargs,
        )


def test_user_message_accepts_images():
    """Test that user messages accept image content (unlike other message types)."""
    # List with images should work
    msg = OpenAIUserMessageParam(
        content=[
            OpenAIChatCompletionContentPartTextParam(text="Describe this image:"),
            OpenAIChatCompletionContentPartImageParam(image_url=OpenAIImageURL(url="http://example.com/image.jpg")),
        ]
    )
    assert len(msg.content) == 2
    assert msg.content[0].text == "Describe this image:"
    assert msg.content[1].image_url.url == "http://example.com/image.jpg"


async def test_convert_message_to_openai_dict_new_user_message():
    """Test convert_message_to_openai_dict_new with UserMessage."""
    message = UserMessage(content="Hello, world!", role="user")
    result = await convert_message_to_openai_dict_new(message)

    assert result["role"] == "user"
    assert result["content"] == "Hello, world!"


async def test_convert_message_to_openai_dict_new_completion_message_with_tool_calls():
    """Test convert_message_to_openai_dict_new with CompletionMessage containing tool calls."""
    message = CompletionMessage(
        content="I'll help you find the weather.",
        tool_calls=[
            ToolCall(
                call_id="call_123",
                tool_name="get_weather",
                arguments='{"city": "Sligo"}',
            )
        ],
        stop_reason=StopReason.end_of_turn,
    )
    result = await convert_message_to_openai_dict_new(message)

    # This would have failed with "Cannot instantiate typing.Union" before the fix
    assert result["role"] == "assistant"
    assert result["content"] == "I'll help you find the weather."
    assert "tool_calls" in result
    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1

    tool_call = result["tool_calls"][0]
    assert tool_call.id == "call_123"
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "Sligo"}'
