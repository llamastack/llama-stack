# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the VertexAI OpenAI ↔ google-genai conversion module.

All google-genai types are mocked via SimpleNamespace — no SDK installation required.
"""

import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llama_stack.providers.remote.inference.vertexai import converters as vertexai_converters
from llama_stack.providers.remote.inference.vertexai.converters import (
    _convert_user_message,
    _extract_logprobs,
    _extract_text_content,
    convert_completion_prompt_to_contents,
    convert_deprecated_function_call_to_tool_choice,
    convert_deprecated_functions_to_tools,
    convert_finish_reason,
    convert_gemini_response_to_openai,
    convert_gemini_response_to_openai_completion,
    convert_gemini_stream_chunk_to_openai,
    convert_gemini_stream_chunk_to_openai_completion,
    convert_model_name,
    convert_openai_messages_to_gemini,
    convert_openai_tools_to_gemini,
    convert_response_format,
    extract_usage,
    generate_completion_id,
)
from llama_stack_api import OpenAICompletion

_convert_image_url_part = getattr(vertexai_converters, "_convert_image_url_part", None)

convert_gemini_response_to_openai = cast(Any, convert_gemini_response_to_openai)
convert_gemini_stream_chunk_to_openai = cast(Any, convert_gemini_stream_chunk_to_openai)

FAKE_IMAGE_BYTES = b"fake image bytes"
FAKE_IMAGE_B64 = base64.b64encode(FAKE_IMAGE_BYTES).decode()


@pytest.fixture
def weather_tool_call() -> dict[str, Any]:
    """Provide a reusable weather tool-call payload."""
    return {
        "id": "call_weather",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}',
        },
    }


def _make_text_part(text: str) -> Any:
    """Build ext part."""
    return SimpleNamespace(text=text, thought=None, function_call=None)


def _make_thought_part(thought_text: str) -> Any:
    """Build hought part."""
    return SimpleNamespace(text=thought_text, thought=True, function_call=None)


def _make_function_call_part(name: str, args: dict) -> Any:
    """Build unction call part."""
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=name, args=args),
    )


def _make_candidate(
    parts: list | None = None,
    finish_reason: str | None = "STOP",
    index: int = 0,
    logprobs_result: Any = None,
) -> Any:
    """Build andidate."""
    content = SimpleNamespace(parts=parts or [])
    return SimpleNamespace(
        content=content,
        finish_reason=finish_reason,
        index=index,
        logprobs_result=logprobs_result,
    )


def _make_response(
    candidates: list | None = None,
    prompt_token_count: int = 10,
    candidates_token_count: int = 20,
    total_token_count: int = 30,
) -> Any:
    """Build esponse."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )
    return SimpleNamespace(candidates=candidates, usage_metadata=usage)


def _make_function_call_response() -> Any:
    """Build unction call response."""
    return _make_response(
        candidates=[
            _make_candidate(
                parts=[_make_function_call_part("get_weather", {"location": "NYC"})],
                finish_reason="STOP",
            )
        ]
    )


def _make_logprob_candidate(
    token: str,
    log_probability: float,
    token_id: int | None = None,
) -> Any:
    """Build ogprob candidate."""
    return SimpleNamespace(token=token, log_probability=log_probability, token_id=token_id)


def _make_top_candidates_entry(candidates: list) -> Any:
    """Build op candidates entry."""
    return SimpleNamespace(candidates=candidates)


def _make_logprobs_result(
    chosen: list | None = None,
    top: list | None = None,
) -> Any:
    """Build ogprobs result."""
    return SimpleNamespace(
        chosen_candidates=chosen or [],
        top_candidates=top or [],
    )


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
        """Test that standard mappings."""
        assert convert_finish_reason(input_reason) == expected

    def test_none(self):
        """Test that none."""
        assert convert_finish_reason(None) == "stop"

    def test_unknown_value(self):
        """Test that unknown value."""
        assert convert_finish_reason("TOTALLY_NEW_REASON") == "stop"

    @pytest.mark.parametrize("input_reason", ["stop", "Stop"])
    def test_case_insensitive(self, input_reason):
        # FinishReason values from SDK are uppercase but let's be defensive
        """Test that case insensitive."""
        assert convert_finish_reason(input_reason) == "stop"


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
        """Test that model name conversion."""
        assert convert_model_name(input_name) == expected


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
        """Test that convert response format."""
        assert convert_response_format(response_format) == expected


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
        """Test that extract text content."""
        assert _extract_text_content(input_content) == expected


class TestConvertImageUrlPart:
    def _convert_part(self, part: dict[str, Any]) -> dict[str, Any] | None:
        """Convert art."""
        assert _convert_image_url_part is not None
        return _convert_image_url_part(part)

    @pytest.mark.parametrize(
        "fmt",
        [
            pytest.param("jpeg", id="jpeg"),
            pytest.param("png", id="png"),
            pytest.param("gif", id="gif"),
            pytest.param("webp", id="webp"),
        ],
    )
    def test_data_uri_to_inline_data(self, fmt):
        """Test that data uri to inline data."""
        part = {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{FAKE_IMAGE_B64}"}}
        result = self._convert_part(part)
        assert result == {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": f"image/{fmt}"}}

    def test_image_detail_parameter_ignored(self):
        """Test that image detail parameter ignored."""
        part = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}",
                "detail": "high",
            },
        }
        result = self._convert_part(part)
        assert result == {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}}

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("file:///path/to/img.png", id="file_scheme"),
            pytest.param("ftp://example.com/img.png", id="ftp_scheme"),
        ],
    )
    def test_unsupported_url_scheme_returns_none(self, url):
        """Test that unsupported url scheme returns none."""
        part = {"type": "image_url", "image_url": {"url": url}}
        assert self._convert_part(part) is None


class TestConvertUserMessageWithImages:
    @pytest.mark.parametrize(
        "message,expected_parts",
        [
            pytest.param({"role": "user", "content": "hello"}, [{"text": "hello"}], id="text_string"),
            pytest.param(
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                [{"text": "hello"}],
                id="text_list",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        }
                    ],
                },
                [{"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}}],
                id="single_image",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        },
                    ],
                },
                [
                    {"text": "hello"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                ],
                id="text_then_image",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        },
                        {"type": "text", "text": "hello"},
                    ],
                },
                [
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                    {"text": "hello"},
                ],
                id="image_then_text",
            ),
            pytest.param(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                        },
                        {"type": "text", "text": "after"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{FAKE_IMAGE_B64}"},
                        },
                    ],
                },
                [
                    {"text": "before"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                    {"text": "after"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/png"}},
                ],
                id="interleaved_text_images",
            ),
        ],
    )
    def test_user_message_conversion(self, message, expected_parts):
        """Test that user message conversion."""
        assert _convert_user_message(message) == {"role": "user", "parts": expected_parts}

    def test_image_only_no_text(self):
        """Test that image only no text."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                }
            ],
        }
        assert _convert_user_message(message) == {
            "role": "user",
            "parts": [{"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}}],
        }

    def test_unsupported_url_scheme_skipped(self):
        """Test that unsupported url scheme skipped."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "file:///path/to/img.png"},
                }
            ],
        }
        assert _convert_user_message(message) == {"role": "user", "parts": []}

    def test_user_message_with_image_in_full_conversion(self):
        """Test that user message with image in full conversion."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{FAKE_IMAGE_B64}"},
                    },
                ],
            }
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert system is None
        assert contents == [
            {
                "role": "user",
                "parts": [
                    {"text": "analyze this image"},
                    {"inline_data": {"data": FAKE_IMAGE_BYTES, "mime_type": "image/jpeg"}},
                ],
            }
        ]


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
        """Test that system and user message conversion."""
        system, contents = convert_openai_messages_to_gemini(messages)
        assert system == expected_system
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_message(self):
        """Test that assistant message."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        system, contents = convert_openai_messages_to_gemini(messages)
        assert len(contents) == 2
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hello there!"}]

    def test_assistant_with_tool_calls(self, weather_tool_call):
        """Test that assistant with tool calls."""
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
        """Test that tool response message."""
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
        """Test that tool response non json."""
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
        """Test that tool response json array wrapped in dict."""
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
        """Test that empty messages."""
        system, contents = convert_openai_messages_to_gemini([])
        assert system is None
        assert contents == []

    def test_assistant_with_text_and_tool_calls(self):
        """Test that assistant with text and tool calls."""
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


class TestConvertOpenAIToolsToGemini:
    def test_single_function_tool(self):
        """Test that single function tool."""
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
        """Test that multiple tools."""
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
        """Test that no convertible tools returns none."""
        assert convert_openai_tools_to_gemini(tools) is None

    def test_tool_without_parameters(self):
        """Test that tool without parameters."""
        tools = [{"type": "function", "function": {"name": "noop", "description": "Does nothing"}}]
        result = convert_openai_tools_to_gemini(tools)
        assert result is not None
        fd = result[0]["function_declarations"][0]
        assert "parameters_json_schema" not in fd


class TestConvertDeprecatedFunctions:
    def test_single_function_converts_to_tool(self):
        """Test that single function converts to tool."""
        functions = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}]
        result = convert_deprecated_functions_to_tools(functions)
        assert result == [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}},
            }
        ]

    def test_multiple_functions(self):
        """Test that multiple functions."""
        functions = [
            {"name": "func_a", "description": "A"},
            {"name": "func_b", "description": "B"},
        ]
        result = convert_deprecated_functions_to_tools(functions)
        assert len(result) == 2
        assert result[0] == {"type": "function", "function": {"name": "func_a", "description": "A"}}
        assert result[1] == {"type": "function", "function": {"name": "func_b", "description": "B"}}

    def test_empty_functions_returns_empty(self):
        """Test that empty functions returns empty."""
        assert convert_deprecated_functions_to_tools([]) == []


class TestConvertDeprecatedFunctionCall:
    def test_auto_passthrough(self):
        """Test that auto passthrough."""
        assert convert_deprecated_function_call_to_tool_choice("auto") == "auto"

    def test_none_passthrough(self):
        """Test that none passthrough."""
        assert convert_deprecated_function_call_to_tool_choice("none") == "none"

    def test_named_function_converts_to_tool_choice(self):
        """Test that named function converts to tool choice."""
        result = convert_deprecated_function_call_to_tool_choice({"name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_unknown_string_passthrough(self):
        """Any unrecognised string passes through unchanged (forward-compat)."""
        assert convert_deprecated_function_call_to_tool_choice("required") == "required"

    def test_dict_without_name_passthrough(self):
        """A dict without a 'name' key is returned as-is (fallback path)."""
        payload: dict[str, Any] = {"mode": "auto"}
        assert convert_deprecated_function_call_to_tool_choice(payload) == payload


class TestConvertGeminiResponseToOpenAI:
    def test_simple_text_response(self):
        """Test that simple text response."""
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
        """Test that function call response sets finish reason and type."""
        response = _make_function_call_response()
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")

        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.type == "function"

    def test_function_call_response_sets_function_payload(self):
        """Test that function call response sets function payload."""
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
        """Test that multi part response."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("Part 1 "), _make_text_part("Part 2")])]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert result.choices[0].message.content == "Part 1 Part 2"

    def test_text_and_function_call_response(self):
        """Test that text and function call response."""
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
        """Test that no candidates safety filtered."""
        response = _make_response(candidates=[])
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"
        assert result.choices[0].message.content is None

    def test_none_candidates(self):
        """Test that none candidates."""
        response: Any = SimpleNamespace(candidates=None, usage_metadata=None)
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "content_filter"

    def test_usage_metadata(self):
        """Test that usage metadata."""
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
        """Test that no usage metadata."""
        response: Any = SimpleNamespace(
            candidates=[_make_candidate(parts=[_make_text_part("hi")])],
            usage_metadata=None,
        )
        result = convert_gemini_response_to_openai(response, "model")
        assert result.usage is None

    def test_empty_parts(self):
        """Test that empty parts."""
        response = _make_response(candidates=[_make_candidate(parts=[])])
        result = convert_gemini_response_to_openai(response, "model")
        assert result.choices[0].message.content is None
        assert result.choices[0].message.tool_calls is None

    def test_candidate_with_none_content(self):
        """Test that candidate with none content."""
        candidate: Any = SimpleNamespace(content=None, finish_reason="STOP", index=0)
        response = _make_response(candidates=[candidate])
        result = convert_gemini_response_to_openai(response, "model")
        assert result.choices[0].message.content is None

    def test_multiple_function_calls(self):
        """Test that multiple function calls."""
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


class TestExtractUsage:
    def test_extract_usage_with_cached_tokens(self):
        """prompt_tokens_details populated when cached_content_token_count is present."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
                cached_content_token_count=30,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 30
        assert usage.completion_tokens_details is None

    def test_extract_usage_with_reasoning_tokens(self):
        """completion_tokens_details populated when thoughts_token_count is present."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
                thoughts_token_count=20,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 20
        assert usage.prompt_tokens_details is None

    def test_extract_usage_no_details_when_fields_absent(self):
        """Both details are None when usage_metadata lacks optional fields."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.prompt_tokens_details is None
        assert usage.completion_tokens_details is None

    def test_extract_usage_with_both_details(self):
        """Both details populated when both optional fields present."""
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
                cached_content_token_count=30,
                thoughts_token_count=20,
            )
        )
        usage = extract_usage(cast(Any, response))
        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.completion_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 30
        assert usage.completion_tokens_details.reasoning_tokens == 20


class TestConvertGeminiStreamChunkToOpenAI:
    def test_first_chunk_has_role(self):
        """Test that first chunk has role."""
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
        """Test that subsequent chunk no role."""
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("lo")], finish_reason=None)])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test123", is_first_chunk=False)
        assert result.choices[0].delta.role is None
        assert result.choices[0].delta.content == "lo"

    def test_final_chunk_with_finish_reason(self):
        """Test that final chunk with finish reason."""
        chunk = _make_response(candidates=[_make_candidate(parts=[_make_text_part("!")], finish_reason="STOP")])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test123", is_first_chunk=False)
        assert result.choices[0].finish_reason == "stop"

    def test_chunk_with_tool_call(self):
        """Test that chunk with tool call."""
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
        """Test that empty chunk no candidates."""
        chunk = _make_response(candidates=[])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-empty", is_first_chunk=True)
        assert len(result.choices) == 1
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].delta.content is None
        assert result.choices[0].finish_reason is None

    def test_stream_usage(self):
        """Test that stream usage."""
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
        """Test that safety filtered chunk."""
        chunk = _make_response(candidates=[_make_candidate(parts=[], finish_reason="SAFETY")])
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-sf", is_first_chunk=False)
        assert result.choices[0].finish_reason == "content_filter"


class TestGenerateCompletionId:
    def test_format(self):
        """Test that format."""
        cid = generate_completion_id()
        assert cid.startswith("chatcmpl-")
        # UUID part should be valid
        uuid_part = cid[len("chatcmpl-") :]
        assert len(uuid_part) == 36  # standard UUID string length

    def test_uniqueness(self):
        """Test that uniqueness."""
        ids = {generate_completion_id() for _ in range(100)}
        assert len(ids) == 100


class TestExtractLogprobs:
    def test_logprobs_none_when_not_present(self):
        """Candidate with no logprobs_result returns None."""
        candidate = _make_candidate(parts=[_make_text_part("hello")])
        result = _extract_logprobs(candidate)
        assert result is None

    def test_logprobs_extracted_from_response(self):
        """Full logprobs extraction from a non-streaming response."""
        chosen = [_make_logprob_candidate("hello", -0.5)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        candidate = _make_candidate(
            parts=[_make_text_part("hello")],
            logprobs_result=logprobs_result,
        )

        result = _extract_logprobs(candidate)
        assert result is not None
        assert result.content is not None
        assert len(result.content) == 1
        assert result.content[0].token == "hello"
        assert result.content[0].logprob == -0.5

    def test_logprobs_bytes_is_utf8_encoding(self):
        """bytes field is UTF-8 encoding of the token string."""
        chosen = [_make_logprob_candidate("hi", -1.0)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        candidate = _make_candidate(logprobs_result=logprobs_result)

        result = _extract_logprobs(candidate)
        assert result is not None
        assert result.content is not None
        expected_bytes = list(b"hi")
        assert result.content[0].bytes == expected_bytes

    def test_top_logprobs_extracted(self):
        """top_logprobs list is populated from top_candidates."""
        chosen = [_make_logprob_candidate("hello", -0.5)]
        top = [
            _make_top_candidates_entry(
                [
                    _make_logprob_candidate("hello", -0.5),
                    _make_logprob_candidate("world", -1.2),
                ]
            )
        ]
        logprobs_result = _make_logprobs_result(chosen=chosen, top=top)
        candidate = _make_candidate(logprobs_result=logprobs_result)

        result = _extract_logprobs(candidate)
        assert result is not None
        assert result.content is not None
        assert result.content[0].top_logprobs is not None
        assert len(result.content[0].top_logprobs) == 2
        assert result.content[0].top_logprobs[0].token == "hello"
        assert result.content[0].top_logprobs[1].token == "world"

    def test_logprobs_in_streaming_chunk(self):
        """Logprobs extraction works in convert_gemini_stream_chunk_to_openai."""
        chosen = [_make_logprob_candidate("hey", -0.3)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_text_part("hey")],
                    finish_reason="STOP",
                    logprobs_result=logprobs_result,
                )
            ]
        )

        result = convert_gemini_stream_chunk_to_openai(
            chunk=chunk,
            model="gemini-2.0-flash",
            completion_id="chatcmpl-test",
            is_first_chunk=True,
        )
        assert result.choices[0].logprobs is not None
        assert result.choices[0].logprobs.content is not None
        assert result.choices[0].logprobs.content[0].token == "hey"


class TestReasoningContent:
    def test_thought_parts_separated_from_text(self):
        """Candidate with both thought and text parts: text goes to text, thought to reasoning_content."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_thought_part("Let me think"), _make_text_part("Answer")])]
        )
        result = convert_gemini_stream_chunk_to_openai(
            response, "gemini-2.5-flash", "chatcmpl-test", is_first_chunk=True
        )
        assert result.choices[0].delta.content == "Answer"
        assert result.choices[0].delta.reasoning_content == "Let me think"

    def test_multiple_thought_parts_concatenated(self):
        """Multiple thinking parts are concatenated into single reasoning_content."""
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_thought_part("Step 1. "),
                        _make_thought_part("Step 2."),
                        _make_text_part("Done"),
                    ]
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.reasoning_content == "Step 1. Step 2."
        assert result.choices[0].delta.content == "Done"

    def test_no_thought_parts_reasoning_content_none(self):
        """Regular candidate without thinking parts has reasoning_content=None."""
        response = _make_response(candidates=[_make_candidate(parts=[_make_text_part("Hello!")])])
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.reasoning_content is None
        assert result.choices[0].delta.content == "Hello!"

    def test_only_thought_parts_text_none(self):
        """Candidate with only thinking parts: text=None, reasoning_content has the thinking text."""
        response = _make_response(candidates=[_make_candidate(parts=[_make_thought_part("I am thinking...")])])
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.content is None
        assert result.choices[0].delta.reasoning_content == "I am thinking..."

    def test_thought_parts_in_streaming_chunk(self):
        """Streaming chunk: delta.reasoning_content is populated from thought parts."""
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_thought_part("Thinking..."), _make_text_part("Result")],
                    finish_reason="STOP",
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(chunk, "model", "chatcmpl-test", is_first_chunk=False)
        assert result.choices[0].delta.reasoning_content == "Thinking..."
        assert result.choices[0].delta.content == "Result"

    def test_thought_parts_in_non_streaming_response(self):
        """Non-streaming response: thinking text is REJOINED into content (no reasoning_content field on message)."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_thought_part("Thinking"), _make_text_part("Answer")])]
        )
        result = convert_gemini_response_to_openai(response, "gemini-2.5-flash")
        # Thinking text is prepended to content for non-streaming
        assert result.choices[0].message.content == "ThinkingAnswer"
        # Message type has no reasoning_content field
        assert not hasattr(result.choices[0].message, "reasoning_content")

    def test_thought_and_tool_calls(self):
        """Thinking parts + function call: reasoning_content has thinking, tool_calls has function call."""
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[
                        _make_thought_part("Let me call the tool"),
                        _make_function_call_part("get_weather", {"city": "NYC"}),
                    ]
                )
            ]
        )
        result = convert_gemini_stream_chunk_to_openai(response, "model", "chatcmpl-test", is_first_chunk=True)
        assert result.choices[0].delta.reasoning_content == "Let me call the tool"
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.content is None


class TestConvertCompletionPromptToContents:
    """Tests for convert_completion_prompt_to_contents()."""

    def test_string_prompt_to_user_message(self):
        """Plain text prompt is wrapped as a single user message."""
        result = convert_completion_prompt_to_contents("hello world")
        assert result == [{"role": "user", "parts": [{"text": "hello world"}]}]

    def test_empty_prompt(self):
        """Empty string prompt still produces a valid user message."""
        result = convert_completion_prompt_to_contents("")
        assert result == [{"role": "user", "parts": [{"text": ""}]}]


class TestConvertGeminiResponseToCompletion:
    """Tests for convert_gemini_response_to_openai_completion()."""

    def test_simple_text_completion(self):
        """Single candidate with text maps to OpenAICompletion correctly."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("The answer is 42")], finish_reason=None)],
        )
        result = convert_gemini_response_to_openai_completion(response, model="gemini-2.5-flash", prompt="What is 42?")
        assert isinstance(result, OpenAICompletion)
        assert result.choices[0].text == "The answer is 42"
        assert result.model == "gemini-2.5-flash"

    def test_no_candidates_returns_empty_text(self):
        """Empty candidates list produces a content_filter choice with empty text."""
        response = _make_response(candidates=[])
        result = convert_gemini_response_to_openai_completion(response, model="gemini-2.5-flash", prompt="hi")
        assert result.choices[0].text == ""
        assert result.choices[0].finish_reason == "content_filter"

    def test_finish_reason_stop(self):
        """STOP finish reason maps to 'stop'."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("done")], finish_reason="STOP")],
        )
        result = convert_gemini_response_to_openai_completion(response, model="gemini-2.5-flash", prompt="hi")
        assert result.choices[0].finish_reason == "stop"

    def test_completion_logprobs_extracted_when_present(self):
        """Logprobs from candidate logprobs_result are converted to OpenAIChoiceLogprobs."""
        chosen = [_make_logprob_candidate("hello", -0.5)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        response = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_text_part("hello")],
                    finish_reason="STOP",
                    logprobs_result=logprobs_result,
                )
            ]
        )
        result = convert_gemini_response_to_openai_completion(response, model="test-model", prompt="hi")
        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert len(logprobs.content) == 1
        assert logprobs.content[0].token == "hello"
        assert logprobs.content[0].logprob == -0.5

    def test_completion_logprobs_none_when_absent(self):
        """Logprobs is None when candidate has no logprobs_result."""
        response = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("hello")], finish_reason="STOP")],
        )
        result = convert_gemini_response_to_openai_completion(response, model="test-model", prompt="hi")
        assert result.choices[0].logprobs is None

    def test_multiple_candidates_return_multiple_choices(self):
        """Test that multiple candidates return multiple choices."""
        response = _make_response(
            candidates=[
                _make_candidate(parts=[_make_text_part("first")], finish_reason="STOP", index=0),
                _make_candidate(parts=[_make_text_part("second")], finish_reason="STOP", index=1),
            ]
        )
        result = convert_gemini_response_to_openai_completion(response, model="test-model", prompt="hi")
        assert len(result.choices) == 2
        assert result.choices[0].text == "first"
        assert result.choices[0].index == 0
        assert result.choices[1].text == "second"
        assert result.choices[1].index == 1


class TestConvertGeminiStreamChunkToOpenAICompletion:
    """Tests for convert_gemini_stream_chunk_to_openai_completion()."""

    def test_stream_completion_chunk_with_text(self):
        """Chunk with a text candidate maps to OpenAICompletion with that text."""
        chunk = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("Hello stream")], finish_reason="STOP")],
        )

        result = convert_gemini_stream_chunk_to_openai_completion(
            chunk, model="gemini-2.5-flash", completion_id="cmpl-abc"
        )

        assert isinstance(result, OpenAICompletion)
        assert result.id == "cmpl-abc"
        assert result.model == "gemini-2.5-flash"
        assert len(result.choices) == 1
        assert result.choices[0].text == "Hello stream"
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].index == 0

    def test_stream_completion_chunk_mid_stream_uses_stop_default(self):
        """Mid-stream chunk with no finish_reason defaults to 'stop' (required field)."""
        chunk = _make_response(
            candidates=[_make_candidate(parts=[_make_text_part("partial")], finish_reason=None)],
        )

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-xyz")

        # finish_reason must be a valid value (not None) since OpenAICompletionChoice requires it
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].text == "partial"

    def test_stream_completion_chunk_with_logprobs(self):
        """Chunk with logprobs_result populates logprobs on the choice."""
        chosen = [_make_logprob_candidate("hello", -0.3)]
        logprobs_result = _make_logprobs_result(chosen=chosen)
        chunk = _make_response(
            candidates=[
                _make_candidate(
                    parts=[_make_text_part("hello")],
                    finish_reason="STOP",
                    logprobs_result=logprobs_result,
                )
            ]
        )

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-lp")

        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert len(logprobs.content) == 1
        assert logprobs.content[0].token == "hello"
        assert logprobs.content[0].logprob == -0.3

    def test_stream_completion_chunk_empty_candidates(self):
        """Chunk with no candidates produces a single fallback choice with empty text."""
        chunk = _make_response(candidates=[])

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-empty")

        assert len(result.choices) == 1
        assert result.choices[0].text == ""
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].index == 0

    def test_stream_completion_chunk_none_candidates(self):
        """Chunk with candidates=None also produces a fallback choice."""
        chunk = _make_response(candidates=None)

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-none")

        assert len(result.choices) == 1
        assert result.choices[0].text == ""
        assert result.choices[0].finish_reason == "stop"

    def test_stream_completion_chunk_multiple_candidates(self):
        """Multiple candidates produce multiple choices with correct indices."""
        chunk = _make_response(
            candidates=[
                _make_candidate(parts=[_make_text_part("first")], finish_reason="STOP", index=0),
                _make_candidate(parts=[_make_text_part("second")], finish_reason="STOP", index=1),
            ]
        )

        result = convert_gemini_stream_chunk_to_openai_completion(chunk, model="test-model", completion_id="cmpl-multi")

        assert len(result.choices) == 2
        assert result.choices[0].text == "first"
        assert result.choices[0].index == 0
        assert result.choices[1].text == "second"
        assert result.choices[1].index == 1
