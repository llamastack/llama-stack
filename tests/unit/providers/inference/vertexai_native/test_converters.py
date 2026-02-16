# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from llama_stack_api import OpenAIChatCompletionRequestWithExtraBody


def build_params(**kwargs: Any) -> OpenAIChatCompletionRequestWithExtraBody:
    payload: dict[str, Any] = {
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(kwargs)
    return OpenAIChatCompletionRequestWithExtraBody.model_validate(payload)


@dataclass
class FakePart:
    kind: str
    payload: dict[str, Any]
    text: str | None = None
    thought: str | None = None
    function_call: Any | None = None

    @classmethod
    def from_text(cls, text: str) -> FakePart:
        return cls(kind="text", payload={"text": text}, text=text)

    @classmethod
    def from_function_call(cls, *, name: str, args: dict[str, Any]) -> FakePart:
        return cls(kind="function_call", payload={"name": name, "args": args})

    @classmethod
    def from_function_response(cls, *, name: str, response: Any) -> FakePart:
        return cls(kind="function_response", payload={"name": name, "response": response})


@dataclass
class FakeContent:
    role: str
    parts: list[FakePart]


@dataclass
class FakeFunctionDeclaration:
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


@dataclass
class FakeTool:
    function_declarations: list[FakeFunctionDeclaration]


class FakeFunctionCallingConfig:
    class Mode:
        AUTO = "AUTO"
        ANY = "ANY"
        NONE = "NONE"

    def __init__(self, *, mode: str, allowed_function_names: list[str] | None = None):
        self.mode = mode
        self.allowed_function_names = allowed_function_names


@dataclass
class FakeToolConfig:
    function_calling_config: FakeFunctionCallingConfig


@dataclass
class FakeThinkingConfig:
    thinking_budget: int | None = None
    include_thoughts: bool | None = None


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_genai_types(monkeypatch):
    import sys

    fake_types = SimpleNamespace(
        Content=FakeContent,
        FunctionCallingConfig=FakeFunctionCallingConfig,
        FunctionDeclaration=FakeFunctionDeclaration,
        GenerateContentConfig=FakeGenerateContentConfig,
        Part=FakePart,
        ThinkingConfig=FakeThinkingConfig,
        Tool=FakeTool,
        ToolConfig=FakeToolConfig,
    )
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(types=fake_types))
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return fake_types


@pytest.fixture
def converters(fake_genai_types):
    return importlib.import_module("llama_stack.providers.remote.inference.vertexai_native.converters")


@pytest.fixture
def vertex_response():
    """Factory for building fake Vertex AI response objects."""

    def _build(
        *,
        parts=None,
        finish_reason="STOP",
        candidates=_SENTINEL,
        prompt_feedback=None,
        prompt_tokens=0,
        completion_tokens=0,
    ):
        if candidates is _SENTINEL:
            candidates = (
                [SimpleNamespace(content=SimpleNamespace(parts=parts or []), finish_reason=finish_reason)]
                if parts is not None
                else None
            )
        usage = (
            SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=completion_tokens)
            if prompt_tokens or completion_tokens
            else None
        )
        resp = SimpleNamespace(candidates=candidates, usage_metadata=usage)
        if prompt_feedback is not None:
            resp.prompt_feedback = prompt_feedback
        return resp

    return _build


_SENTINEL = object()


@pytest.mark.parametrize(
    ("role", "content", "expected_role", "expected_kind"),
    [
        ("user", "hello", "user", "text"),
        ("assistant", "answer", "model", "text"),
        ("developer", "guide", None, None),
        ("tool", "ok", "user", "function_response"),
        ("system", "instructions", None, None),
    ],
)
def test_convert_messages_role_mapping(converters, role, content, expected_role, expected_kind):
    """Verify OpenAI roles map to google-genai roles and part types correctly."""
    msg = {"role": role, "content": content, "tool_call_id": "call_1"}
    if role == "tool":
        msg["name"] = "get_weather"
    system_instruction, converted = converters.convert_messages([msg])

    if role in ("system", "developer"):
        assert system_instruction == content
        assert converted == []
        return

    assert system_instruction is None
    assert len(converted) == 1
    assert converted[0].role == expected_role
    assert converted[0].parts[0].kind == expected_kind


def test_convert_messages_supports_text_part_arrays(converters):
    """Verify content arrays with multiple text parts produce multiple Part.from_text entries."""
    system_instruction, converted = converters.convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
    )

    assert system_instruction is None
    assert converted[0].parts == [FakePart.from_text("hello"), FakePart.from_text("world")]


def test_convert_messages_multiple_system_messages_are_concatenated(converters):
    """Verify multiple system messages are joined with double newlines into one instruction."""
    system_instruction, converted = converters.convert_messages(
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b"},
        ]
    )

    assert system_instruction == "a\n\nb"
    assert len(converted) == 1
    assert converted[0].role == "user"


@pytest.mark.parametrize(
    "unsupported_part",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        {"type": "file", "file": {"file_id": "file-1"}},
    ],
)
def test_convert_messages_rejects_non_text_parts(converters, unsupported_part):
    """Verify non-text content parts (image_url, file) raise ValueError."""
    with pytest.raises(ValueError, match="Only text parts are supported"):
        converters.convert_messages([{"role": "user", "content": [unsupported_part]}])


@pytest.mark.parametrize(
    ("tool_choice", "expected_mode"),
    [
        ("auto", "AUTO"),
        ("required", "ANY"),
        ("none", "NONE"),
    ],
)
def test_convert_tools_tool_choice_modes(converters, tool_choice, expected_mode):
    """Verify OpenAI tool_choice strings map to google-genai FunctionCallingConfig modes."""
    converted_tools, tool_config = converters.convert_tools(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice=tool_choice,
    )

    assert converted_tools is not None
    assert len(converted_tools) == 1
    assert converted_tools[0].function_declarations[0].name == "weather"
    assert converted_tools[0].function_declarations[0].description == "Get weather"
    assert converted_tools[0].function_declarations[0].parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert tool_config is not None
    assert tool_config.function_calling_config.mode == expected_mode


def test_convert_tools_returns_none_when_no_inputs(converters):
    """Verify None tools and tool_choice produce None outputs."""
    converted_tools, tool_config = converters.convert_tools(tools=None, tool_choice=None)

    assert converted_tools is None
    assert tool_config is None


@pytest.mark.parametrize(
    ("description", "param_kwargs", "sys_instruction", "tools", "tool_config", "assertions"),
    [
        (
            "maps_openai_params",
            {"temperature": 0.3, "max_tokens": 128, "top_p": 0.75, "stop": ["END"]},
            "sys",
            lambda: [FakeTool(function_declarations=[])],
            lambda: FakeToolConfig(function_calling_config=FakeFunctionCallingConfig(mode="AUTO")),
            {
                "temperature": 0.3,
                "max_output_tokens": 128,
                "top_p": 0.75,
                "stop_sequences": ["END"],
            },
        ),
        (
            "prefers_max_completion_tokens",
            {"max_tokens": 32, "max_completion_tokens": 64},
            None,
            lambda: None,
            lambda: None,
            {"max_output_tokens": 64},
        ),
        (
            "passes_model_extra_fields",
            {
                "thinking": {"thinking_budget": 2048},
                "cached_content": "projects/p/locations/l/cachedContents/abc",
                "response_mime_type": "application/json",
            },
            None,
            lambda: None,
            lambda: None,
            {
                "cached_content": "projects/p/locations/l/cachedContents/abc",
                "response_mime_type": "application/json",
            },
        ),
    ],
    ids=["maps_openai_params", "prefers_max_completion_tokens", "passes_model_extra_fields"],
)
def test_build_generate_config(converters, description, param_kwargs, sys_instruction, tools, tool_config, assertions):
    """Verify OpenAI request params map to google-genai GenerateContentConfig fields."""
    params = build_params(**param_kwargs)
    config = converters.build_generate_config(
        params=params,
        system_instruction=sys_instruction,
        tools=tools(),
        tool_config=tool_config(),
    )

    for attr, expected in assertions.items():
        assert getattr(config, attr) == expected, f"{attr}: {getattr(config, attr)} != {expected}"

    if sys_instruction:
        assert config.system_instruction == sys_instruction
        assert config.tools is not None
        assert config.tool_config is not None

    if "thinking" in param_kwargs:
        assert config.thinking_config.thinking_budget == 2048


@pytest.mark.parametrize(
    ("request_kwargs", "expected"),
    [
        ({"logprobs": True}, ["logprobs"]),
        ({"logit_bias": {"12": 1.0}}, ["logit_bias"]),
        ({"n": 2}, ["n"]),
        ({"frequency_penalty": 0.1, "presence_penalty": 0.2}, ["frequency_penalty", "presence_penalty"]),
        ({"n": 1}, []),
    ],
)
def test_collect_ignored_params(converters, request_kwargs, expected):
    """Unsupported OpenAI params are collected for warning, except harmless defaults."""
    params = build_params(**request_kwargs)

    assert converters.collect_ignored_params(params) == expected


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
        ("RECITATION", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "stop"),
        ("SOME_OTHER_REASON", "stop"),
    ],
)
def test_convert_finish_reason(converters, finish_reason, expected):
    """Maps google-genai finish reasons to OpenAI equivalents."""
    assert converters.convert_finish_reason(finish_reason) == expected


def test_convert_response_maps_text_finish_reason_and_usage(converters, vertex_response):
    """Non-streaming response extracts text, finish reason, and token usage."""
    response = vertex_response(
        parts=[SimpleNamespace(text="hello", thought=None, function_call=None)],
        finish_reason="STOP",
        prompt_tokens=3,
        completion_tokens=5,
    )

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_1")

    assert converted.id == "req_1"
    assert converted.model == "google/gemini-2.5-flash"
    assert converted.choices[0].message.content == "hello"
    assert converted.choices[0].finish_reason == "stop"
    assert converted.usage is not None
    assert converted.usage.prompt_tokens == 3
    assert converted.usage.completion_tokens == 5
    assert converted.usage.total_tokens == 8


def test_convert_response_prompt_level_block_returns_content_filter(converters, vertex_response):
    """When Vertex AI blocks at the prompt level (no candidates), finish_reason is content_filter."""
    response = vertex_response(
        candidates=[],
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        prompt_tokens=5,
    )

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_block")

    assert converted.choices[0].finish_reason == "content_filter"
    assert converted.choices[0].message.content == "Request blocked by content filter: SAFETY"


def test_convert_response_prompt_level_block_without_reason(converters, vertex_response):
    """When Vertex AI blocks at the prompt level without providing a reason, content is None."""
    response = vertex_response(candidates=None)

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_block2")

    assert converted.choices[0].finish_reason == "content_filter"
    assert converted.choices[0].message.content is None


def test_convert_response_excludes_thought_from_content(converters, vertex_response):
    """Thought parts are separated from text content in non-streaming responses."""
    response = vertex_response(
        parts=[
            SimpleNamespace(text="Let me think...", thought=True, function_call=None),
            SimpleNamespace(text="The answer is 42.", thought=None, function_call=None),
        ],
    )

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_think")

    assert converted.choices[0].message.content == "The answer is 42."


def test_convert_response_preserves_empty_string_text(converters, vertex_response):
    """Empty-string text parts are preserved (not dropped by falsy check)."""
    response = vertex_response(
        parts=[
            SimpleNamespace(text="hello", thought=None, function_call=None),
            SimpleNamespace(text="", thought=None, function_call=None),
            SimpleNamespace(text="world", thought=None, function_call=None),
        ],
    )

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_empty")

    assert converted.choices[0].message.content == "hello\n\nworld"


def test_convert_response_maps_tool_calls(converters, vertex_response):
    """Function call parts in response become OpenAI-format tool_calls."""
    response = vertex_response(
        parts=[
            SimpleNamespace(
                text=None,
                thought=None,
                function_call=SimpleNamespace(name="weather", args={"city": "Paris"}),
            )
        ],
        finish_reason="MAX_TOKENS",
        prompt_tokens=10,
        completion_tokens=20,
    )

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_2")

    assert converted.choices[0].finish_reason == "length"
    assert converted.choices[0].message.tool_calls is not None
    assert converted.choices[0].message.tool_calls[0].function.name == "weather"
    assert converted.choices[0].message.tool_calls[0].function.arguments == json.dumps({"city": "Paris"})


def test_convert_response_tool_calls_overrides_stop_to_tool_calls(converters, vertex_response):
    """When tool calls are present and Vertex returns STOP, finish_reason becomes 'tool_calls'."""
    response = vertex_response(
        parts=[
            SimpleNamespace(
                text=None,
                thought=None,
                function_call=SimpleNamespace(name="search", args={"q": "test"}),
            )
        ],
        finish_reason="STOP",
    )

    converted = converters.convert_response(response=response, model="google/gemini-2.5-flash", request_id="req_tc")

    assert converted.choices[0].finish_reason == "tool_calls"
    assert converted.choices[0].message.tool_calls is not None


def test_convert_stream_chunk_tool_calls_overrides_stop_to_tool_calls(converters):
    """Streaming chunks with tool calls and STOP finish_reason become 'tool_calls'."""
    chunk = SimpleNamespace(
        text=None,
        function_calls=[SimpleNamespace(name="search", args={"q": "test"})],
        thought=None,
        usage_metadata=None,
        finish_reason="STOP",
    )

    converted = converters.convert_stream_chunk(
        chunk=chunk, model="google/gemini-2.5-flash", request_id="req_tc_stream", chunk_index=0
    )

    assert converted.choices[0].finish_reason == "tool_calls"
    assert converted.choices[0].delta.tool_calls is not None


@pytest.mark.parametrize(
    ("chunk", "expected_content", "expected_reasoning", "expect_tool_calls"),
    [
        (SimpleNamespace(text="hello", function_calls=None, thought=None, usage_metadata=None), "hello", None, False),
        (
            SimpleNamespace(
                text=None,
                function_calls=[SimpleNamespace(name="weather", args={"city": "Paris"})],
                thought=None,
                usage_metadata=None,
            ),
            None,
            None,
            True,
        ),
        (
            SimpleNamespace(text=None, function_calls=None, thought="thinking", usage_metadata=None),
            None,
            "thinking",
            False,
        ),
    ],
)
def test_convert_stream_chunk_maps_delta_fields(
    converters, chunk, expected_content, expected_reasoning, expect_tool_calls
):
    """Streaming chunks map text, function_calls, and thought to delta fields."""
    converted = converters.convert_stream_chunk(
        chunk=chunk,
        model="google/gemini-2.5-flash",
        request_id="req_3",
        chunk_index=0,
    )

    delta = converted.choices[0].delta
    assert delta.content == expected_content
    assert delta.reasoning_content == expected_reasoning
    if expect_tool_calls:
        assert delta.tool_calls is not None
        assert delta.tool_calls[0].function.name == "weather"
    else:
        assert delta.tool_calls is None


def test_convert_stream_chunk_maps_usage_only_when_present(converters):
    """Token usage is included in stream chunks only when usage_metadata is present."""
    chunk = SimpleNamespace(
        text=None,
        function_calls=None,
        thought=None,
        usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=11),
    )

    converted = converters.convert_stream_chunk(
        chunk=chunk,
        model="google/gemini-2.5-flash",
        request_id="req_4",
        chunk_index=5,
    )

    assert converted.usage is not None
    assert converted.usage.prompt_tokens == 7
    assert converted.usage.completion_tokens == 11
    assert converted.usage.total_tokens == 18


def test_convert_messages_assistant_tool_calls_become_function_call_parts(converters):
    """Assistant tool_calls become function_call parts, tool responses use the mapped function name."""
    system_instruction, converted = converters.convert_messages(
        [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 25°C", "tool_call_id": "call_abc"},
        ]
    )

    assert system_instruction is None
    assert len(converted) == 3

    assistant_msg = converted[1]
    assert assistant_msg.role == "model"
    assert len(assistant_msg.parts) == 1
    assert assistant_msg.parts[0].kind == "function_call"
    assert assistant_msg.parts[0].payload["name"] == "get_weather"
    assert assistant_msg.parts[0].payload["args"] == {"city": "Paris"}

    tool_msg = converted[2]
    assert tool_msg.role == "user"
    assert tool_msg.parts[0].kind == "function_response"
    assert tool_msg.parts[0].payload["name"] == "get_weather"


def test_convert_messages_none_content_produces_no_text_parts(converters):
    """Messages with content=None produce only function_call parts, no text parts."""
    system_instruction, converted = converters.convert_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            },
        ]
    )

    assert system_instruction is None
    assert len(converted) == 1
    parts = converted[0].parts
    assert all(p.kind == "function_call" for p in parts)
    assert not any(p.kind == "text" for p in parts)


def test_convert_messages_tool_response_uses_explicit_name_field(converters):
    """When tool message has a 'name' field, it takes priority over tool_call_id mapping."""
    system_instruction, converted = converters.convert_messages(
        [
            {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "explicit_fn"},
        ]
    )

    assert system_instruction is None
    assert len(converted) == 1
    assert converted[0].parts[0].payload["name"] == "explicit_fn"


def test_convert_messages_tool_response_raises_on_missing_name(converters):
    """When tool message has no name field and no prior assistant tool_call, raise ValueError."""
    with pytest.raises(ValueError, match="Cannot resolve function name"):
        converters.convert_messages(
            [
                {"role": "tool", "content": "result", "tool_call_id": "call_orphan"},
            ]
        )


def test_convert_messages_invalid_tool_call_json_raises(converters):
    """Invalid JSON in tool call arguments raises ValueError with function name context."""
    with pytest.raises(ValueError, match="Invalid JSON in tool call arguments for 'get_weather'"):
        converters.convert_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{not valid json}"},
                        }
                    ],
                },
            ]
        )


def test_convert_tools_dict_tool_choice_maps_to_any_with_allowed_names(converters):
    """Dict-based tool_choice with a specific function maps to mode=ANY with allowed_function_names."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}

    converted_tools, tool_config = converters.convert_tools(tools=tools, tool_choice=tool_choice)

    assert converted_tools is not None
    assert tool_config is not None
    assert tool_config.function_calling_config.mode == "ANY"
    assert tool_config.function_calling_config.allowed_function_names == ["get_weather"]


# ── Fix #6/#7: build_generate_config validation ──────────────────────


def test_build_generate_config_rejects_invalid_thinking_keys(converters):
    """Unknown keys in thinking dict raise ValueError listing valid options."""
    params = build_params(thinking={"bad_key": True})
    with pytest.raises(ValueError, match="Unknown keys in 'thinking'"):
        converters.build_generate_config(params=params, system_instruction=None, tools=None, tool_config=None)


def test_build_generate_config_rejects_non_dict_thinking(converters):
    """Non-dict thinking value raises TypeError."""
    params = build_params(thinking="enabled")
    with pytest.raises(TypeError, match="thinking.*must be a dict"):
        converters.build_generate_config(params=params, system_instruction=None, tools=None, tool_config=None)


def test_build_generate_config_rejects_non_string_cached_content(converters):
    """Non-string cached_content raises TypeError."""
    params = build_params(cached_content=12345)
    with pytest.raises(TypeError, match="cached_content.*must be a string"):
        converters.build_generate_config(params=params, system_instruction=None, tools=None, tool_config=None)


def test_build_generate_config_rejects_non_string_response_mime_type(converters):
    """Non-string response_mime_type raises TypeError."""
    params = build_params(response_mime_type=["application/json"])
    with pytest.raises(TypeError, match="response_mime_type.*must be a string"):
        converters.build_generate_config(params=params, system_instruction=None, tools=None, tool_config=None)


# ── Fix #8: max_output_tokens zero handling ──────────────────────────


def test_build_generate_config_max_completion_tokens_takes_precedence(converters):
    """max_completion_tokens takes precedence over max_tokens when both are set."""
    params = build_params(max_completion_tokens=1, max_tokens=100)
    config = converters.build_generate_config(params=params, system_instruction=None, tools=None, tool_config=None)

    assert config.max_output_tokens == 1


# ── Fix #9: streaming first chunk role ───────────────────────────────


def test_convert_stream_chunk_first_chunk_includes_role(converters):
    """First streaming chunk (index 0) includes role='assistant' in delta."""
    chunk = SimpleNamespace(text="hello", function_calls=None, thought=None, usage_metadata=None)
    converted = converters.convert_stream_chunk(
        chunk=chunk, model="google/gemini-2.5-flash", request_id="req_role", chunk_index=0
    )

    assert converted.choices[0].delta.role == "assistant"


def test_convert_stream_chunk_subsequent_chunks_omit_role(converters):
    """Subsequent streaming chunks (index > 0) do not include role in delta."""
    chunk = SimpleNamespace(text="world", function_calls=None, thought=None, usage_metadata=None)
    converted = converters.convert_stream_chunk(
        chunk=chunk, model="google/gemini-2.5-flash", request_id="req_role2", chunk_index=3
    )

    assert converted.choices[0].delta.role is None


# ── Fix #10: _as_mapping prefers to_dict/model_dump ──────────────────


def test_as_mapping_prefers_to_dict_over_dict(converters):
    """_as_mapping prefers to_dict() over __dict__ when both are available."""

    class FakeSDKObject:
        def __init__(self):
            self.raw = "should_not_appear"

        def to_dict(self):
            return {"from_sdk": True}

    result = converters._as_mapping(FakeSDKObject())
    assert result == {"from_sdk": True}


def test_as_mapping_prefers_model_dump_over_dict(converters):
    """_as_mapping prefers model_dump() (Pydantic) over __dict__."""

    class FakePydantic:
        def __init__(self):
            self.raw = "should_not_appear"

        def model_dump(self, *, exclude_none=False):
            return {"from_pydantic": True}

    result = converters._as_mapping(FakePydantic())
    assert result == {"from_pydantic": True}
