# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from llama_stack_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
    OpenAIFinishReason,
)

_ROLE_MAP = {
    "user": "user",
    "assistant": "model",
}

_TOOL_CHOICE_MODE_MAP = {
    "auto": "AUTO",
    "required": "ANY",
    "none": "NONE",
}

_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "RECITATION": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
}

_IGNORED_PARAM_NAMES = (
    "logprobs",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
)


def _get_genai_types() -> Any:
    """Lazily import and return the ``google.genai.types`` module."""
    return importlib.import_module("google.genai.types")


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a value into a plain dict, handling Pydantic models and dataclasses."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return cast(dict[str, Any], value.to_dict())
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    return {}


def _normalize_text_content(content: Any) -> list[str]:
    """Extract text strings from an OpenAI message ``content`` field."""
    if content is None:
        return []

    if isinstance(content, str):
        return [content]

    if not isinstance(content, Sequence):
        return [str(content)]

    segments: list[str] = []
    for part in content:
        part_dict = _as_mapping(part)
        part_type = part_dict.get("type")
        if part_type != "text":
            raise ValueError(f"Only text parts are supported in vertexai_native v1, got '{part_type}'")
        segments.append(str(part_dict.get("text", "")))
    return segments


def _build_usage(usage_metadata: Any) -> OpenAIChatCompletionUsage | None:
    """Translate google-genai usage metadata to OpenAI usage format."""
    if usage_metadata is None:
        return None
    usage_dict = _as_mapping(usage_metadata)
    prompt_tokens = int(usage_dict.get("prompt_token_count", 0) or 0)
    completion_tokens = int(usage_dict.get("candidates_token_count", 0) or 0)
    return OpenAIChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _convert_function_call(function_call: Any, index: int) -> OpenAIChatCompletionToolCall:
    """Convert a single google-genai function call into an OpenAI tool call."""
    function_call_dict = _as_mapping(function_call)
    call_id = str(function_call_dict.get("id") or f"call_{index}")
    name = str(function_call_dict.get("name") or "")
    args = function_call_dict.get("args")
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args if args is not None else {})

    return OpenAIChatCompletionToolCall(
        index=index,
        id=call_id,
        type="function",
        function=OpenAIChatCompletionToolCallFunction(name=name, arguments=arguments),
    )


def collect_ignored_params(params: OpenAIChatCompletionRequestWithExtraBody) -> list[str]:
    """Return request fields intentionally ignored by the native converter."""

    ignored: list[str] = []
    payload = params.model_dump(exclude_none=False)

    for key in _IGNORED_PARAM_NAMES:
        if payload.get(key) is not None:
            ignored.append(key)

    if (payload.get("n") or 0) > 1:
        ignored.append("n")

    if payload.get("function_call") is not None:
        ignored.append("function_call")
    if payload.get("functions") is not None:
        ignored.append("functions")

    return ignored


def _build_tool_call_id_map(messages: Sequence[Any]) -> dict[str, str]:
    """Build tool_call_id → function_name map from assistant tool_calls.

    Tool response messages reference calls by opaque ID, but the google-genai
    SDK needs the actual function name in ``Part.from_function_response``.
    """
    mapping: dict[str, str] = {}
    for message in messages:
        msg = _as_mapping(message)
        for tc in msg.get("tool_calls") or []:
            tc_dict = _as_mapping(tc)
            tc_id = tc_dict.get("id")
            fn = _as_mapping(tc_dict.get("function"))
            if tc_id and fn.get("name"):
                mapping[str(tc_id)] = str(fn["name"])
    return mapping


def _parse_tool_args(raw: Any, fn_name: str) -> dict[str, Any]:
    """Parse tool call arguments, handling invalid JSON gracefully."""
    if not isinstance(raw, str):
        return raw if raw is not None else {}
    try:
        return cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool call arguments for '{fn_name}': {raw[:200]}") from e


def _convert_tool_call_parts(tool_calls: Sequence[Any]) -> list[Any]:
    """Convert OpenAI-format tool_calls into google-genai function_call Parts."""
    types = _get_genai_types()
    parts: list[Any] = []
    for tc in tool_calls:
        tc_dict = _as_mapping(tc)
        fn = _as_mapping(tc_dict.get("function"))
        fn_name = str(fn.get("name", ""))
        fn_args_raw = fn.get("arguments", "{}")
        fn_args = _parse_tool_args(fn_args_raw, fn_name)
        parts.append(types.Part.from_function_call(name=fn_name, args=fn_args))
    return parts


def _convert_tool_message(
    message_dict: Mapping[str, Any],
    text_segments: list[str],
    tool_call_id_to_name: dict[str, str],
) -> Any:
    """Convert an OpenAI tool-result message into a google-genai Content with a function_response Part."""
    types = _get_genai_types()
    tool_call_id = str(message_dict.get("tool_call_id", ""))

    # Resolve function name: explicit name field > mapping from prior assistant tool_calls
    fn_name = message_dict.get("name") or tool_call_id_to_name.get(tool_call_id)
    if not fn_name:
        raise ValueError(
            f"Cannot resolve function name for tool response (tool_call_id={tool_call_id!r}). "
            "Include the prior assistant message with tool_calls or set the 'name' field."
        )

    part = types.Part.from_function_response(
        name=fn_name,
        response={"content": "\n".join(text_segments)},
    )
    return types.Content(role="user", parts=[part])


def _convert_chat_message(
    role: str,
    message_dict: Mapping[str, Any],
    text_segments: list[str],
) -> Any | None:
    """Convert an OpenAI user/assistant message into a google-genai Content object."""
    types = _get_genai_types()
    genai_role = _ROLE_MAP.get(role)
    if genai_role is None:
        raise ValueError(f"Unsupported message role: {role}")

    parts = [types.Part.from_text(text=text_segment) for text_segment in text_segments]
    if role == "assistant":
        parts.extend(_convert_tool_call_parts(message_dict.get("tool_calls") or []))

    return types.Content(role=genai_role, parts=parts) if parts else None


def convert_messages(messages: Sequence[Any]) -> tuple[str | None, list[Any]]:
    """Convert OpenAI chat messages into Vertex native system/content payloads."""

    system_segments: list[str] = []
    contents: list[Any] = []
    tool_call_id_to_name = _build_tool_call_id_map(messages)

    for message in messages:
        message_dict = _as_mapping(message)
        role = str(message_dict.get("role", ""))
        text_segments = _normalize_text_content(message_dict.get("content"))

        if role in ("system", "developer"):
            system_segments.append("\n".join(text_segments))
        elif role == "tool":
            contents.append(_convert_tool_message(message_dict, text_segments, tool_call_id_to_name))
        else:
            content = _convert_chat_message(role, message_dict, text_segments)
            if content:
                contents.append(content)

    system_instruction = "\n\n".join(system_segments) if system_segments else None
    return system_instruction, contents


def _convert_tool_declarations(tools: list[dict[str, Any]]) -> list[Any] | None:
    """Convert OpenAI function tool schemas into google-genai Tool declarations."""
    types = _get_genai_types()
    declarations = []
    for tool in tools:
        tool_dict = _as_mapping(tool)
        if tool_dict.get("type") != "function":
            continue
        function_def = _as_mapping(tool_dict.get("function"))
        declaration = types.FunctionDeclaration(
            name=function_def.get("name"),
            description=function_def.get("description"),
            parameters=function_def.get("parameters"),
        )
        declarations.append(types.Tool(function_declarations=[declaration]))
    return declarations or None


def _convert_tool_choice(tool_choice: str | dict[str, Any]) -> Any:
    """Convert an OpenAI tool_choice value into a google-genai ToolConfig."""
    types = _get_genai_types()

    if isinstance(tool_choice, str):
        mode_name = _TOOL_CHOICE_MODE_MAP.get(tool_choice)
        if mode_name is None:
            raise ValueError(f"Unsupported tool_choice: {tool_choice}")
        mode = getattr(types.FunctionCallingConfig.Mode, mode_name)
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))

    choice_dict = _as_mapping(tool_choice)
    if choice_dict.get("type") == "function":
        fn = _as_mapping(choice_dict.get("function"))
        fn_name = fn.get("name")
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=[fn_name] if fn_name else None,
            )
        )

    raise ValueError(f"Unsupported tool_choice: {tool_choice}")


def convert_tools(
    tools: list[dict[str, Any]] | None, tool_choice: str | dict[str, Any] | None
) -> tuple[list[Any] | None, Any]:
    """Convert OpenAI tool schemas and tool choice into google-genai tool types."""

    converted_tools = _convert_tool_declarations(tools) if tools else None
    tool_config = _convert_tool_choice(tool_choice) if tool_choice is not None else None
    return converted_tools, tool_config


_THINKING_CONFIG_KEYS = frozenset({"thinking_budget", "include_thoughts"})


def _build_thinking_config(types: Any, value: Any) -> Any | None:
    """Validate and build a ThinkingConfig from a user-supplied dict."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"'thinking' must be a dict, got {type(value).__name__}")
    unknown = set(value) - _THINKING_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in 'thinking': {sorted(unknown)}")
    return types.ThinkingConfig(**value)


def _validated_str(value: Any, name: str) -> str | None:
    """Return *value* if it is ``None`` or a ``str``, else raise ``TypeError``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a string, got {type(value).__name__}")
    return value


def build_generate_config(
    params: OpenAIChatCompletionRequestWithExtraBody,
    system_instruction: str | None,
    tools: list[Any] | None,
    tool_config: Any,
) -> Any:
    """Build a GenerateContentConfig from OpenAI request parameters."""

    types = _get_genai_types()

    stop_sequences: list[str] | None
    if isinstance(params.stop, str):
        stop_sequences = [params.stop]
    else:
        stop_sequences = params.stop

    max_output_tokens = params.max_completion_tokens if params.max_completion_tokens is not None else params.max_tokens

    extra = params.model_extra or {}
    thinking_config = _build_thinking_config(types, extra.get("thinking"))
    cached_content = _validated_str(extra.get("cached_content"), "cached_content")
    response_mime_type = _validated_str(extra.get("response_mime_type"), "response_mime_type")

    return types.GenerateContentConfig(
        temperature=params.temperature,
        max_output_tokens=max_output_tokens,
        top_p=params.top_p,
        stop_sequences=stop_sequences,
        system_instruction=system_instruction,
        tools=tools,
        tool_config=tool_config,
        thinking_config=thinking_config,
        cached_content=cached_content,
        response_mime_type=response_mime_type,
    )


def convert_finish_reason(reason: Any) -> OpenAIFinishReason:
    """Map google-genai finish reasons to OpenAI-compatible reason strings."""

    normalized = str(getattr(reason, "name", reason)).upper()
    return cast(OpenAIFinishReason, _FINISH_REASON_MAP.get(normalized, "stop"))


def _resolve_finish_reason(reason: Any, has_tool_calls: bool) -> OpenAIFinishReason:
    """Determine the final finish reason, promoting to ``tool_calls`` when appropriate."""

    finish_reason = convert_finish_reason(reason)
    if has_tool_calls and finish_reason == "stop":
        return cast(OpenAIFinishReason, "tool_calls")
    return finish_reason


def _extract_parts(
    candidate_dict: dict[str, Any],
) -> tuple[list[str], list[OpenAIChatCompletionToolCall], list[str]]:
    """Extract text segments, tool calls, and thought segments from candidate parts."""

    content_dict = _as_mapping(candidate_dict.get("content"))
    parts = content_dict.get("parts") or []

    text_segments: list[str] = []
    tool_calls: list[OpenAIChatCompletionToolCall] = []
    thought_segments: list[str] = []
    for index, part in enumerate(parts):
        part_dict = _as_mapping(part)
        thought = part_dict.get("thought")
        if thought:
            thought_text = part_dict.get("text")
            if thought_text is not None:
                thought_segments.append(str(thought_text))
            continue
        text = part_dict.get("text")
        if text is not None:
            text_segments.append(str(text))
        function_call = part_dict.get("function_call")
        if function_call is not None:
            tool_calls.append(_convert_function_call(function_call, index))

    for index, function_call in enumerate(candidate_dict.get("function_calls") or [], start=len(tool_calls)):
        tool_calls.append(_convert_function_call(function_call, index))

    return text_segments, tool_calls, thought_segments


def _build_blocked_response(response: Any, model: str, request_id: str) -> OpenAIChatCompletion:
    """Build an OpenAI response for a prompt blocked before candidate generation."""

    block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    message = OpenAIChatCompletionResponseMessage(
        role="assistant",
        content=f"Request blocked by content filter: {block_reason}" if block_reason else None,
    )
    choice = OpenAIChoice(index=0, message=message, finish_reason="content_filter")
    return OpenAIChatCompletion(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=_build_usage(getattr(response, "usage_metadata", None)),
    )


def convert_response(response: Any, model: str, request_id: str) -> OpenAIChatCompletion:
    """Convert a non-streaming google-genai response into OpenAI chat format."""

    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    if candidate is None:
        return _build_blocked_response(response, model, request_id)

    candidate_dict = _as_mapping(candidate)
    text_segments, tool_calls, thought_segments = _extract_parts(candidate_dict)

    # NOTE: OpenAIChatCompletionResponseMessage does not have a reasoning_content
    # field (only OpenAIChoiceDelta does, for streaming). Thought content from
    # Gemini's thinking mode is extracted but cannot be surfaced in non-streaming
    # responses until the schema adds support.
    message = OpenAIChatCompletionResponseMessage(
        role="assistant",
        content="\n".join(text_segments) if text_segments else None,
        tool_calls=tool_calls or None,
    )

    choice = OpenAIChoice(
        index=0,
        message=message,
        finish_reason=_resolve_finish_reason(candidate_dict.get("finish_reason"), bool(tool_calls)),
    )

    return OpenAIChatCompletion(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=_build_usage(getattr(response, "usage_metadata", None)),
    )


def convert_stream_chunk(chunk: Any, model: str, request_id: str, chunk_index: int) -> OpenAIChatCompletionChunk:
    """Convert one streaming google-genai chunk into OpenAI chunk format."""

    tool_calls: list[OpenAIChatCompletionToolCall] | None = None
    function_calls = getattr(chunk, "function_calls", None)
    if function_calls:
        tool_calls = [_convert_function_call(fc, i) for i, fc in enumerate(function_calls)]

    delta = OpenAIChoiceDelta(
        role="assistant" if chunk_index == 0 else None,
        content=getattr(chunk, "text", None),
        tool_calls=tool_calls,
        reasoning_content=getattr(chunk, "thought", None),
    )
    finish_reason = (
        _resolve_finish_reason(chunk.finish_reason, bool(tool_calls))
        if getattr(chunk, "finish_reason", None) is not None
        else None
    )

    choice = OpenAIChunkChoice(
        delta=delta,
        index=0,
        finish_reason=finish_reason,
    )

    return OpenAIChatCompletionChunk(
        id=request_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=_build_usage(getattr(chunk, "usage_metadata", None)),
    )
