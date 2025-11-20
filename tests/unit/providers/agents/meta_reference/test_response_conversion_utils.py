# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from unittest.mock import AsyncMock

import pytest

from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    _extract_citations_from_text,
    convert_chat_choice_to_response_message,
    convert_response_content_to_chat_content,
    convert_response_input_to_chat_messages,
    convert_response_text_to_chat_response_format,
    get_message_type_by_role,
    is_function_tool_call,
)
from llama_stack_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIDeveloperMessageParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIResponseFormatText,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack_api.openai_responses import (
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)


@pytest.fixture
def mock_files_api():
    """Mock files API for testing."""
    return AsyncMock()


class TestConvertChatChoiceToResponseMessage:
    async def test_convert_string_content(self):
        choice = OpenAIChoice(
            message=OpenAIAssistantMessageParam(content="Test message"),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(choice)

        assert result.role == "assistant"
        assert result.status == "completed"
        assert len(result.content) == 1
        assert isinstance(result.content[0], OpenAIResponseOutputMessageContentOutputText)
        assert result.content[0].text == "Test message"

    async def test_convert_text_param_content(self):
        choice = OpenAIChoice(
            message=OpenAIAssistantMessageParam(
                content=[OpenAIChatCompletionContentPartTextParam(text="Test text param")]
            ),
            finish_reason="stop",
            index=0,
        )

        with pytest.raises(ValueError) as exc_info:
            await convert_chat_choice_to_response_message(choice)

        assert "does not yet support output content type" in str(exc_info.value)


class TestConvertResponseContentToChatContent:
    async def test_convert_string_content(self, mock_files_api):
        result = await convert_response_content_to_chat_content("Simple string", mock_files_api)
        assert result == "Simple string"

    async def test_convert_text_content_parts(self, mock_files_api):
        content = [
            OpenAIResponseInputMessageContentText(text="First part"),
            OpenAIResponseOutputMessageContentOutputText(text="Second part"),
        ]

        result = await convert_response_content_to_chat_content(content, mock_files_api)

        assert len(result) == 2
        assert isinstance(result[0], OpenAIChatCompletionContentPartTextParam)
        assert result[0].text == "First part"
        assert isinstance(result[1], OpenAIChatCompletionContentPartTextParam)
        assert result[1].text == "Second part"

    async def test_convert_image_content(self, mock_files_api):
        content = [OpenAIResponseInputMessageContentImage(image_url="https://example.com/image.jpg", detail="high")]

        result = await convert_response_content_to_chat_content(content, mock_files_api)

        assert len(result) == 1
        assert isinstance(result[0], OpenAIChatCompletionContentPartImageParam)
        assert result[0].image_url.url == "https://example.com/image.jpg"
        assert result[0].image_url.detail == "high"


class TestConvertResponseInputToChatMessages:
    async def test_convert_string_input(self):
        result = await convert_response_input_to_chat_messages("User message")

        assert len(result) == 1
        assert isinstance(result[0], OpenAIUserMessageParam)
        assert result[0].content == "User message"

    async def test_convert_function_tool_call_output(self):
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_123",
                name="test_function",
                arguments='{"param": "value"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output="Tool output",
                call_id="call_123",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 2
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert result[0].tool_calls[0].id == "call_123"
        assert result[0].tool_calls[0].function.name == "test_function"
        assert result[0].tool_calls[0].function.arguments == '{"param": "value"}'
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == "Tool output"
        assert result[1].tool_call_id == "call_123"

    async def test_convert_function_tool_call(self):
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_456",
                name="test_function",
                arguments='{"param": "value"}',
            )
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 1
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "call_456"
        assert result[0].tool_calls[0].function.name == "test_function"
        assert result[0].tool_calls[0].function.arguments == '{"param": "value"}'

    async def test_convert_function_call_ordering(self):
        input_items = [
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_123",
                name="test_function_a",
                arguments='{"param": "value"}',
            ),
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_456",
                name="test_function_b",
                arguments='{"param": "value"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output="AAA",
                call_id="call_123",
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                output="BBB",
                call_id="call_456",
            ),
        ]

        result = await convert_response_input_to_chat_messages(input_items)
        assert len(result) == 4
        assert isinstance(result[0], OpenAIAssistantMessageParam)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "call_123"
        assert result[0].tool_calls[0].function.name == "test_function_a"
        assert result[0].tool_calls[0].function.arguments == '{"param": "value"}'
        assert isinstance(result[1], OpenAIToolMessageParam)
        assert result[1].content == "AAA"
        assert result[1].tool_call_id == "call_123"
        assert isinstance(result[2], OpenAIAssistantMessageParam)
        assert len(result[2].tool_calls) == 1
        assert result[2].tool_calls[0].id == "call_456"
        assert result[2].tool_calls[0].function.name == "test_function_b"
        assert result[2].tool_calls[0].function.arguments == '{"param": "value"}'
        assert isinstance(result[3], OpenAIToolMessageParam)
        assert result[3].content == "BBB"
        assert result[3].tool_call_id == "call_456"

    async def test_convert_response_message(self):
        input_items = [
            OpenAIResponseMessage(
                role="user",
                content=[OpenAIResponseInputMessageContentText(text="User text")],
            )
        ]

        result = await convert_response_input_to_chat_messages(input_items)

        assert len(result) == 1
        assert isinstance(result[0], OpenAIUserMessageParam)
        # Content should be converted to chat content format
        assert len(result[0].content) == 1
        assert result[0].content[0].text == "User text"


class TestConvertResponseTextToChatResponseFormat:
    async def test_convert_text_format(self):
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text"))
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatText)
        assert result.type == "text"

    async def test_convert_json_object_format(self):
        text = OpenAIResponseText(format={"type": "json_object"})
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatJSONObject)

    async def test_convert_json_schema_format(self):
        schema_def = {"type": "object", "properties": {"test": {"type": "string"}}}
        text = OpenAIResponseText(
            format={
                "type": "json_schema",
                "name": "test_schema",
                "schema": schema_def,
            }
        )
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatJSONSchema)
        assert result.json_schema["name"] == "test_schema"
        assert result.json_schema["schema"] == schema_def

    async def test_default_text_format(self):
        text = OpenAIResponseText()
        result = await convert_response_text_to_chat_response_format(text)

        assert isinstance(result, OpenAIResponseFormatText)
        assert result.type == "text"


class TestGetMessageTypeByRole:
    async def test_user_role(self):
        result = await get_message_type_by_role("user")
        assert result == OpenAIUserMessageParam

    async def test_system_role(self):
        result = await get_message_type_by_role("system")
        assert result == OpenAISystemMessageParam

    async def test_assistant_role(self):
        result = await get_message_type_by_role("assistant")
        assert result == OpenAIAssistantMessageParam

    async def test_developer_role(self):
        result = await get_message_type_by_role("developer")
        assert result == OpenAIDeveloperMessageParam

    async def test_unknown_role(self):
        result = await get_message_type_by_role("unknown")
        assert result is None


class TestIsFunctionToolCall:
    def test_is_function_tool_call_true(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=OpenAIChatCompletionToolCallFunction(
                name="test_function",
                arguments="{}",
            ),
        )
        tools = [
            OpenAIResponseInputToolFunction(
                type="function", name="test_function", parameters={"type": "object", "properties": {}}
            ),
            OpenAIResponseInputToolWebSearch(type="web_search"),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is True

    def test_is_function_tool_call_false_different_name(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=OpenAIChatCompletionToolCallFunction(
                name="other_function",
                arguments="{}",
            ),
        )
        tools = [
            OpenAIResponseInputToolFunction(
                type="function", name="test_function", parameters={"type": "object", "properties": {}}
            ),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is False

    def test_is_function_tool_call_false_no_function(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=None,
        )
        tools = [
            OpenAIResponseInputToolFunction(
                type="function", name="test_function", parameters={"type": "object", "properties": {}}
            ),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is False

    def test_is_function_tool_call_false_wrong_type(self):
        tool_call = OpenAIChatCompletionToolCall(
            index=0,
            id="call_123",
            function=OpenAIChatCompletionToolCallFunction(
                name="web_search",
                arguments="{}",
            ),
        )
        tools = [
            OpenAIResponseInputToolWebSearch(type="web_search"),
        ]

        result = is_function_tool_call(tool_call, tools)
        assert result is False


class TestExtractCitationsFromText:
    def test_extract_citations_and_annotations(self):
        text = "Start [not-a-file]. New source <|file-abc123|>. "
        text += "Other source <|file-def456|>? Repeat source <|file-abc123|>! No citation."
        file_mapping = {"file-abc123": "doc1.pdf", "file-def456": "doc2.txt"}

        annotations, cleaned_text = _extract_citations_from_text(text, file_mapping)

        expected_annotations = [
            OpenAIResponseAnnotationFileCitation(file_id="file-abc123", filename="doc1.pdf", index=30),
            OpenAIResponseAnnotationFileCitation(file_id="file-def456", filename="doc2.txt", index=44),
            OpenAIResponseAnnotationFileCitation(file_id="file-abc123", filename="doc1.pdf", index=59),
        ]
        expected_clean_text = "Start [not-a-file]. New source. Other source? Repeat source! No citation."

        assert cleaned_text == expected_clean_text
        assert annotations == expected_annotations
        # OpenAI cites at the end of the sentence
        assert cleaned_text[expected_annotations[0].index] == "."
        assert cleaned_text[expected_annotations[1].index] == "?"
        assert cleaned_text[expected_annotations[2].index] == "!"
