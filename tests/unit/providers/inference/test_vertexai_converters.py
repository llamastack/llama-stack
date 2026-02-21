# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the VertexAI OpenAI ↔ google-genai conversion module.

All google-genai types are mocked via SimpleNamespace — no SDK installation required.
"""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llama_stack.providers.remote.inference.vertexai.converters import (
    _extract_text_content,
    convert_finish_reason,
    convert_gemini_response_to_openai,
    convert_gemini_stream_chunk_to_openai,
    convert_model_name,
    convert_openai_messages_to_gemini,
    convert_openai_tools_to_gemini,
    convert_response_format,
    generate_completion_id,
)

convert_gemini_response_to_openai = cast(Any, convert_gemini_response_to_openai)
convert_gemini_stream_chunk_to_openai = cast(Any, convert_gemini_stream_chunk_to_openai)


@pytest.fixture
def weather_tool_call() -> dict[str, Any]:
    return {
        "id": "call_weather",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}',
        },
    }


# ---------------------------------------------------------------------------
# Helpers to build mock Gemini response objects
# ---------------------------------------------------------------------------


def _make_text_part(text: str) -> Any:
    return SimpleNamespace(text=text, function_call=None)


def _make_function_call_part(name: str, args: dict) -> Any:
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=name, args=args),
    )


def _make_candidate(
    parts: list | None = None,
    finish_reason: str | None = "STOP",
    index: int = 0,
) -> Any:
    content = SimpleNamespace(parts=parts or [])
    return SimpleNamespace(content=content, finish_reason=finish_reason, index=index)


def _make_response(
    candidates: list | None = None,
    prompt_token_count: int = 10,
    candidates_token_count: int = 20,
    total_token_count: int = 30,
) -> Any:
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )
    return SimpleNamespace(candidates=candidates, usage_metadata=usage)


def _make_function_call_response() -> Any:
    return _make_response(
        candidates=[
            _make_candidate(
                parts=[_make_function_call_part("get_weather", {"location": "NYC"})],
                finish_reason="STOP",
            )
        ]
    )


# ===================================================================
# convert_finish_reason
# ===================================================================


class TestConvertFinishReason:
    @pytest.mark.parametrize(
        "input_reason,expected",
        [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
            ("LANGUAGE", "content_filter"),
            ("BLOCKLIST", "content_filter"),
            ("PROHIBITED_CONTENT", "content_filter"),
            ("SPII", "content_filter"),
            ("MALFORMED_FUNCTION_CALL", "stop"),
            ("OTHER", "stop"),
        ],
    )
    def test_standard_mappings(self, input_reason, expected):
        assert convert_finish_reason(input_reason) == expected

    def test_none(self):
        assert convert_finish_reason(None) == "stop"

    def test_unknown_value(self):
        assert convert_finish_reason("TOTALLY_NEW_REASON") == "stop"

    @pytest.mark.parametrize("input_reason", ["stop", "Stop"])
    def test_case_insensitive(self, input_reason):
        # FinishReason values from SDK are uppercase but let's be defensive
        assert convert_finish_reason(input_reason) == "stop"


# ===================================================================
# convert_model_name
# ===================================================================


class TestConvertModelName:
    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("google/gemini-2.5-flash", "gemini-2.5-flash"),
            ("gemini-2.5-flash", "gemini-2.5-flash"),
            ("meta/llama-3", "meta/llama-3"),
            ("", ""),
            ("google/", ""),
        ],
    )
    def test_model_name_conversion(self, input_name, expected):
        assert convert_model_name(input_name) == expected


# ===================================================================
# convert_response_format
# ===================================================================


class TestConvertResponseFormat:
    @pytest.mark.parametrize(
        "response_format,expected",
        [
            (None, {}),
            ({"type": "text"}, {}),
            ({"type": "json_object"}, {"response_mime_type": "application/json"}),
            (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    },
                },
                {
                    "response_mime_type": "application/json",
                    "response_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            ),
            ({"type": "json_schema", "json_schema": {"name": "test"}}, {"response_mime_type": "application/json"}),
            ({"type": "unknown"}, {}),
        ],
    )
    def test_convert_response_format(self, response_format, expected):
        assert convert_response_format(response_format) == expected


# ===================================================================
# _extract_text_content
# ===================================================================


class TestExtractTextContent:
    @pytest.mark.parametrize(
        "input_content,expected",
        [
            ("hello", "hello"),
            (None, ""),
            ([], ""),
            (
                [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
                "hello world",
            ),
            (
                [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
                "hello",
            ),
        ],
    )
    def test_extract_text_content(self, input_content, expected):
        assert _extract_text_content(input_content) == expected


# ===================================================================
# convert_openai_messages_to_gemini
# ===================================================================


class TestConvertOpenAIMessagesToGemini:
    @pytest.mark.parametrize(
        "messages,expected_system",
        [
            ([{"role": "user", "content": "Hello"}], None),
            (
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "You are helpful.",
            ),
            (
                [
                    {"role": "system", "content": "Rule 1."},
                    {"role": "system", "content": "Rule 2."},
                    {"role": "user", "content": "Hi"},
                ],
                "Rule 1.\nRule 2.",
            ),
            (
                [
                    {"role": "developer", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                ],
                "Be concise.",
            ),
        ],
    )
    def test_system_and_user_message_conversion(self, messages, expected_system):
        system, contents = convert_openai_messages_to_gemini(messages)
        assert system == expected_system
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_message(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 2
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hello there!"}]

    def test_assistant_with_tool_calls(self, weather_tool_call):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{**weather_tool_call, "id": "call_123"}],
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 2
        model_msg = contents[1]
        assert model_msg["role"] == "model"
        assert len(model_msg["parts"]) == 1
        fc = model_msg["parts"][0]["function_call"]
        assert fc["name"] == "get_weather"
        assert fc["args"] == {"location": "Boston"}

    def test_tool_response_message(self, weather_tool_call):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        **weather_tool_call,
                        "id": "call_abc",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temperature": 72}',
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 3
        tool_msg = contents[2]
        assert tool_msg["role"] == "user"
        fr = tool_msg["parts"][0]["function_response"]
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"temperature": 72}

    def test_tool_response_non_json(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "some_tool", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_xyz",
                "content": "plain text result",
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        tool_msg = contents[2]
        fr = tool_msg["parts"][0]["function_response"]
        assert fr["response"] == {"result": "plain text result"}

    def test_tool_response_json_array_wrapped_in_dict(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_arr",
                        "type": "function",
                        "function": {"name": "list_items", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_arr",
                "content": "[1, 2, 3]",
            },
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        tool_msg = contents[2]
        fr = tool_msg["parts"][0]["function_response"]
        assert fr["response"] == {"result": [1, 2, 3]}

    def test_empty_messages(self):
        system, contents = convert_openai_messages_to_gemini([])
        assert system is None
        assert contents == []

    def test_assistant_with_text_and_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        model_msg = contents[0]
        assert model_msg["role"] == "model"
        # Should have both text and function_call parts
        assert len(model_msg["parts"]) == 2
        assert model_msg["parts"][0] == {"text": "Let me check."}
        assert "function_call" in model_msg["parts"][1]

    def test_tool_call_id_not_found(self):
        """When tool_call_id doesn't match any assistant message, use 'unknown' as name."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "nonexistent",
                "content": "result",
            }
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        fr = contents[0]["parts"][0]["function_response"]
        assert fr["name"] == "unknown"


# ===================================================================
# convert_openai_tools_to_gemini
# ===================================================================


class TestConvertOpenAIToolsToGemini:
    def test_single_function_tool(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        assert len(result) == 1
        fds = result[0]["function_declarations"]
        assert len(fds) == 1
        assert fds[0]["name"] == "get_weather"
        assert fds[0]["description"] == "Get current weather"
        assert "properties" in fds[0]["parameters_json_schema"]

    def test_multiple_tools(self):
        tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A"}},
            {"type": "function", "function": {"name": "tool_b", "description": "B"}},
        ]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        fds = result[0]["function_declarations"]
        assert len(fds) == 2
        assert fds[0]["name"] == "tool_a"
        assert fds[1]["name"] == "tool_b"

    @pytest.mark.parametrize("tools", [None, [], [{"type": "code_interpreter", "other": "data"}]])
    def test_no_convertible_tools_returns_none(self, tools):
        assert convert_openai_tools_to_gemini(tools) is None

    def test_tool_without_parameters(self):
        tools = [{"type": "function", "function": {"name": "noop", "description": "Does nothing"}}]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        fd = result[0]["function_declarations"][0]
        assert "parameters_json_schema" not in fd


# ===================================================================
# convert_gemini_response_to_openai
# ===================================================================


class TestConvertGeminiResponseToOpenAI:
    def test_simple_text_response(self):
        response = _make_response(candidates=[_make_candidate(parts=[_make_text_part("Hello!")])])
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        assert result.object == "chat.completion"
        assert result.model == "gemini-2.5-flash"
        assert result.id.startswith("chatcmpl-")
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.tool_calls is None

    def test_function_call_response_sets_finish_reason_and_type(self):
        response = _make_function_call_response()
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.type == "function"

    def test_function_call_response_sets_function_payload(self):
        response = _make_function_call_response()
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        tc = tool_calls[0]
        assert tc.function is not None
        assert tc.function.name == "get_weather"
        assert tc.function.arguments is not None
        assert json.loads(tc.function.arguments) == {"location": "NYC"}
        assert tc.id is not None
        assert tc.id.startswith("call_")

    def test_multi_part_response(self):
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("Part 1 "), _make_text_part("Part 2")])]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.choices[0].message.content == "Part 1 Part 2"

    def test_text_and_function_call_response(self):
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_text_part("Let me check."),
                        _make_function_call_part("search", {"q": "test"}),
                    ]
                )
            ]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.choices[0].message.content == "Let me check."
        assert result.choices[0].finish_reason == "tool_calls"
        tc_list = result.choices[0].message.tool_calls
        assert tc_list is not None
        assert len(tc_list) == 1

    def test_no_candidates_safety_filtered(self):
        response = _make_response(candidates=[])
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"
        assert result.choices[0].message.content is None

    def test_none_candidates(self):
        response: Any = SimpleNamespace(candidates=None, usage_metadata=None)
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"

    def test_usage_metadata(self):
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("hi")])],
            prompt_token_count=15,
            candidates_token_count=5,
            total_token_count=20,
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.usage is not None
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 20

    def test_no_usage_metadata(self):
        response: Any = SimpleNamespace(
            candidates=[_make_candidate(parts=[_make_text_part("hi")])],
            usage_metadata=None,
        )
        result = convert_gemini_response_to_openai(response, "model")
        assert result.usage is None

    def test_empty_parts(self):
        response = _make_response(candidates=[_make_candidate(parts=[])])
        result = convert_gemini_response_to_openai(response, "model")
        assert result.choices[0].message.content is None
        assert result.choices[0].message.tool_calls is None

    def test_candidate_with_none_content(self):
        candidate: Any = SimpleNamespace(content=None, finish_reason="STOP", index=0)
        response = _make_response(candidates=[candidate])
        result = convert_gemini_response_to_openai(response, "model")
        assert result.choices[0].message.content is None

    def test_multiple_function_calls(self):
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_function_call_part("func_a", {"x": 1}),
                        _make_function_call_part("func_b", {"y": 2}),
                    ]
                )
            ]
        )
        result = convert_gemini_response_to_openai(response, "model")
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].index == 0
        assert tool_calls[1].index == 1
        assert tool_calls[0].function is not None
        assert tool_calls[0].function.name == "func_a"
        assert tool_calls[1].function is not None
        assert tool_calls[1].function.name == "func_b"


# ===================================================================
# convert_gemini_stream_chunk_to_openai
# ===================================================================


class TestConvertGeminiStreamChunkToOpenAI:
    def test_first_chunk_has_role(self):
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("Hel")], finish_reason=None)])
        result = convert_gemini_stream_chunk_to_openai(
            chunk, "gemini-2.5-flash", "chatcmpl-test123", is_first_chunk=True
        )
        assert result.object == "chat.completion.chunk"
        assert result.id == "chatcmpl-test123"
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].delta.content == "Hel"
        assert result.choices[0].finish_reason is None

    def test_subsequent_chunk_no_role(self):
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("lo")], finish_reason=None)])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test123", is_first_chunk=False)
        assert result.choices[0].delta.role is None
        assert result.choices[0].delta.content == "lo"

    def test_final_chunk_with_finish_reason(self):
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("!")], finish_reason="STOP")])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test123", is_first_chunk=False)
        assert result.choices[0].finish_reason == "stop"

    def test_chunk_with_tool_call(self):
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_function_call_part("search", {"q": "test"})],
                    finish_reason=None,
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-tc", is_first_chunk=True)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].finish_reason == "tool_calls"

    def test_empty_chunk_no_candidates(self):
        chunk = _make_response(candidates=[])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-empty", is_first_chunk=True)
        assert len(result.choices) == 1
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].delta.content is None
        assert result.choices[0].finish_reason is None

    def test_stream_usage(self):
        chunk = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("done")], finish_reason="STOP")],
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-u", is_first_chunk=False)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_safety_filtered_chunk(self):
        chunk = _make_response(candidates=[_make_candidate(parts=[], finish_reason="SAFETY")])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-sf", is_first_chunk=False)
        assert result.choices[0].finish_reason == "content_filter"


# ===================================================================
# _generate_completion_id
# ===================================================================


class TestGenerateCompletionId:
    def test_format(self):
        cid = generate_completion_id()
        assert cid.startswith("chatcmpl-")
        # UUID part should be valid
        uuid_part = cid[len("chatcmpl-") :]
        assert len(uuid_part) == 36  # standard UUID string length

    def test_uniqueness(self):
        ids = {generate_completion_id() for _ in range(100)}
        assert len(ids) == 100
