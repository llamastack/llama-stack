# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the BuiltinInteractionsImpl translation logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.providers.inline.interactions.config import InteractionsConfig
from llama_stack.providers.inline.interactions.impl import BuiltinInteractionsImpl
from llama_stack_api.interactions.models import (
    GoogleCreateInteractionRequest,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleTextContent,
)


def _msg_to_dict(msg):
    """Convert a Pydantic message model to dict for easy assertion."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return dict(msg)


@pytest.fixture
def impl():
    mock_inference = AsyncMock()
    return BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference)


class TestRequestTranslation:
    def test_simple_string_input(self, impl):
        request = GoogleCreateInteractionRequest(
            model="gemini-2.5-flash",
            input="Hello",
        )
        result = impl._google_to_openai(request)

        assert result.model == "gemini-2.5-flash"
        assert len(result.messages) == 1
        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "user"
        assert m["content"] == "Hello"

    def test_conversation_turns(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(role="user", content=[GoogleTextContent(text="Question 1")]),
                GoogleInputTurn(role="model", content=[GoogleTextContent(text="Answer 1")]),
                GoogleInputTurn(role="user", content=[GoogleTextContent(text="Question 2")]),
            ],
        )
        result = impl._google_to_openai(request)

        assert len(result.messages) == 3
        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        m2 = _msg_to_dict(result.messages[2])
        assert m0["role"] == "user"
        assert m0["content"] == "Question 1"
        assert m1["role"] == "assistant"
        assert m1["content"] == "Answer 1"
        assert m2["role"] == "user"
        assert m2["content"] == "Question 2"

    def test_model_role_mapped_to_assistant(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(role="model", content=[GoogleTextContent(text="I am the model")]),
            ],
        )
        result = impl._google_to_openai(request)

        m = _msg_to_dict(result.messages[0])
        assert m["role"] == "assistant"

    def test_system_instruction(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            system_instruction="You are helpful.",
        )
        result = impl._google_to_openai(request)

        assert len(result.messages) == 2
        m0 = _msg_to_dict(result.messages[0])
        m1 = _msg_to_dict(result.messages[1])
        assert m0["role"] == "system"
        assert m0["content"] == "You are helpful."
        assert m1["role"] == "user"

    def test_generation_config_temperature(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(temperature=0.7),
        )
        result = impl._google_to_openai(request)
        assert result.temperature == 0.7

    def test_generation_config_top_p(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(top_p=0.9),
        )
        result = impl._google_to_openai(request)
        assert result.top_p == 0.9

    def test_generation_config_max_output_tokens(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(max_output_tokens=500),
        )
        result = impl._google_to_openai(request)
        assert result.max_tokens == 500

    def test_generation_config_top_k_extra_body(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            generation_config=GoogleGenerationConfig(top_k=40),
        )
        result = impl._google_to_openai(request)
        assert result.model_extra.get("top_k") == 40

    def test_stream_flag(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input="Hi",
            stream=True,
        )
        result = impl._google_to_openai(request)
        assert result.stream is True

    def test_multi_content_turn(self, impl):
        request = GoogleCreateInteractionRequest(
            model="m",
            input=[
                GoogleInputTurn(
                    role="user",
                    content=[
                        GoogleTextContent(text="Line 1"),
                        GoogleTextContent(text="Line 2"),
                    ],
                ),
            ],
        )
        result = impl._google_to_openai(request)

        m = _msg_to_dict(result.messages[0])
        assert m["content"] == "Line 1\nLine 2"


class TestResponseTranslation:
    def test_simple_text_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hello!"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 5

        result = impl._openai_to_google(openai_resp, "gemini-2.5-flash")

        assert result.id.startswith("interaction-")
        assert result.status == "completed"
        assert len(result.outputs) == 1
        assert result.outputs[0].type == "text"
        assert result.outputs[0].text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    def test_empty_response(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = None
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 5
        openai_resp.usage.completion_tokens = 0

        result = impl._openai_to_google(openai_resp, "m")

        assert result.status == "completed"
        assert len(result.outputs) == 0

    def test_missing_usage(self, impl):
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "Hi"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = None

        result = impl._openai_to_google(openai_resp, "m")

        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 0


class TestStreamingTranslation:
    async def test_text_streaming(self, impl):
        chunks = []

        for i, text in enumerate(["Hello", " world", "!"]):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop" if i == 2 else None
            chunk.usage = None
            chunks.append(chunk)

        async def mock_stream():
            for c in chunks:
                yield c

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m"):
            events.append(event)

        assert events[0].event_type == "interaction.start"
        assert events[0].id.startswith("interaction-")
        assert events[1].event_type == "content.start"
        assert events[1].index == 0
        assert events[2].event_type == "content.delta"
        assert events[2].delta.text == "Hello"
        assert events[3].event_type == "content.delta"
        assert events[3].delta.text == " world"
        assert events[4].event_type == "content.delta"
        assert events[4].delta.text == "!"
        assert events[5].event_type == "content.stop"
        assert events[6].event_type == "interaction.complete"
        assert events[6].status == "completed"

    async def test_streaming_with_usage(self, impl):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hi"
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None

        # Usage-only chunk
        chunk2 = MagicMock()
        chunk2.choices = []
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 10
        chunk2.usage.completion_tokens = 5

        async def mock_stream():
            yield chunk1
            yield chunk2

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m"):
            events.append(event)

        complete_event = [e for e in events if e.event_type == "interaction.complete"][0]
        assert complete_event.usage.input_tokens == 10
        assert complete_event.usage.output_tokens == 5
        assert complete_event.usage.total_tokens == 15

    async def test_empty_streaming(self, impl):
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            chunk.usage = None
            yield chunk

        events = []
        async for event in impl._stream_openai_to_google(mock_stream(), "m"):
            events.append(event)

        assert events[0].event_type == "interaction.start"
        # No content.start/delta/stop since no content
        assert events[1].event_type == "interaction.complete"
