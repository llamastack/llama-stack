# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.models.llama.datatypes import RawTextItem
from llama_stack.providers.utils.inference.prompt_adapter import (
    convert_openai_message_to_raw_message,
)


class TestConvertOpenAIMessageToRawMessage:
    """Test conversion of OpenAI message types to RawMessage format."""

    async def test_user_message_conversion(self):
        msg = OpenAIUserMessageParam(role="user", content="Hello world")
        raw_msg = await convert_openai_message_to_raw_message(msg)

        assert raw_msg.role == "user"
        assert isinstance(raw_msg.content, RawTextItem)
        assert raw_msg.content.text == "Hello world"

    async def test_assistant_message_conversion(self):
        msg = OpenAIAssistantMessageParam(role="assistant", content="Hi there!")
        raw_msg = await convert_openai_message_to_raw_message(msg)

        assert raw_msg.role == "assistant"
        assert isinstance(raw_msg.content, RawTextItem)
        assert raw_msg.content.text == "Hi there!"
        assert raw_msg.tool_calls == []
