# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pure translation functions between OpenAI format and google-genai native API.

No SDK calls or side effects — only type conversions.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from llama_stack.log import get_logger
from llama_stack_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
)
from llama_stack_api.inference.models import (
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
    OpenAIFinishReason,
)

logger = get_logger(__name__, category="inference")

if TYPE_CHECKING:
    from google.genai import types as genai_types

_GEMINI_TO_OPENAI_FINISH_REASON: dict[str, OpenAIFinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "LANGUAGE": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
}


def convert_finish_reason(
    finish_reason: str | None,
) -> OpenAIFinishReason:
    """Map a Gemini FinishReason string to the OpenAI finish_reason literal."""
    if finish_reason is None:
        return "stop"
    reason_str = str(finish_reason).upper()
    return _GEMINI_TO_OPENAI_FINISH_REASON.get(reason_str, "stop")


def convert_model_name(model: str) -> str:
    """Strip the ``google/`` prefix that llama-stack prepends to Gemini model IDs.

    Example: ``"google/gemini-2.5-flash"`` → ``"gemini-2.5-flash"``
    """
    if model.startswith("google/"):
        return model[len("google/") :]
    return model


def convert_response_format(
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert an OpenAI ``response_format`` parameter to google-genai config kwargs.

    Returns a dict that can be merged into ``GenerateContentConfig`` kwargs.
    Supports ``json_object`` → ``response_mime_type='application/json'``
    and ``json_schema`` → ``response_mime_type='application/json'`` + ``response_schema``.
    """
    if response_format is None:
        return {}

    fmt_type = response_format.get("type")
    if fmt_type == "json_object":
        return {"response_mime_type": "application/json"}
    if fmt_type == "json_schema":
        result: dict[str, Any] = {"response_mime_type": "application/json"}
        json_schema = response_format.get("json_schema")
        if json_schema and isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if schema:
                result["response_schema"] = schema
        return result
    # "text" or unknown → no special config
    return {}


def _extract_text_content(content: str | list[dict[str, Any]] | None) -> str:
    """Extract plain text from OpenAI message content (string or content parts list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # content is a list of content parts
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
        elif isinstance(part, str):
            texts.append(part)
        elif hasattr(part, "type") and part.type == "text":
            texts.append(getattr(part, "text", ""))
    return "".join(texts)


def _convert_user_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI user message to a Gemini Content dict."""
    content = msg.get("content")
    text = _extract_text_content(content)
    if isinstance(content, list):
        # TODO: convert image/audio/video parts to Gemini inline_data or
        # file_data parts instead of discarding them — the native SDK
        # supports multimodal input.
        non_text = [p for p in content if isinstance(p, dict) and p.get("type") not in ("text", None)]
        if non_text:
            logger.warning(
                "Non-text content parts (e.g. images) in user message were discarded; "
                "only text is forwarded to Gemini. Dropped %d part(s).",
                len(non_text),
            )
    return {"role": "user", "parts": [{"text": text}]}


def _parse_tool_call_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments from string or dict form."""
    if not isinstance(arguments, str):
        return arguments if isinstance(arguments, dict) else {}
    try:
        parsed: dict[str, Any] = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed


def _convert_assistant_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI assistant message to a Gemini Content dict.

    Returns ``None`` when the message has no text content and no tool calls.
    """
    parts: list[dict[str, Any]] = []

    text = _extract_text_content(msg.get("content"))
    if text:
        parts.append({"text": text})

    for tc in msg.get("tool_calls") or []:
        if hasattr(tc, "model_dump"):
            tc = tc.model_dump()
        func = tc.get("function", {})
        parts.append(
            {
                "function_call": {
                    "name": func.get("name", ""),
                    "args": _parse_tool_call_arguments(func.get("arguments", "{}")),
                }
            }
        )

    return {"role": "model", "parts": parts} if parts else None


def _convert_tool_message(msg: dict[str, Any], all_messages: list[Any]) -> dict[str, Any]:
    """Convert an OpenAI tool-result message to a Gemini Content dict."""
    tool_call_id = msg.get("tool_call_id", "")
    tool_content = _extract_text_content(msg.get("content"))
    try:
        response_data = json.loads(tool_content)
    except (json.JSONDecodeError, TypeError):
        response_data = {"result": tool_content}
    if not isinstance(response_data, dict):
        response_data = {"result": response_data}

    func_name = _find_function_name_for_tool_call_id(all_messages, tool_call_id)
    return {
        "role": "user",
        "parts": [{"function_response": {"name": func_name, "response": response_data}}],
    }


def _normalize_message(msg: Any) -> dict[str, Any]:
    """Ensure *msg* is a plain dict (handles Pydantic models)."""
    if hasattr(msg, "model_dump"):
        result: dict[str, Any] = msg.model_dump()
        return result
    return dict(msg)


def convert_openai_messages_to_gemini(
    messages: list[Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Gemini Content dicts.

    Returns ``(system_instruction, contents)`` where:
    - ``system_instruction`` is extracted from system/developer messages (or ``None``).
    - ``contents`` is a list of Gemini ``Content``-like dicts with ``role`` and ``parts``.

    Gemini uses ``"user"`` and ``"model"`` roles (no ``"assistant"``).
    Tool results use ``"user"`` role with ``function_response`` parts.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for raw_msg in messages:
        msg = _normalize_message(raw_msg)
        role = msg.get("role", "")

        if role in ("system", "developer"):
            text = _extract_text_content(msg.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            contents.append(_convert_user_message(msg))
        elif role == "assistant":
            converted = _convert_assistant_message(msg)
            if converted:
                contents.append(converted)
        elif role == "tool":
            contents.append(_convert_tool_message(msg, messages))

    system_instruction = "\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _find_function_name_for_tool_call_id(messages: list[Any], tool_call_id: str) -> str:
    """Search through messages for the function name matching a tool_call_id."""
    for msg in messages:
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, "model_dump"):
                        tc = tc.model_dump()
                    if tc.get("id") == tool_call_id:
                        func = tc.get("function", {})
                        return str(func.get("name", "unknown"))
    return "unknown"


def convert_openai_tools_to_gemini(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI tools array to Gemini Tool format.

    OpenAI format::

        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]

    Gemini format::

        [{"function_declarations": [{"name": ..., "description": ..., "parameters": {...}}]}]

    Returns ``None`` if no tools are provided.
    """
    if not tools:
        return None

    function_declarations: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        decl: dict[str, Any] = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if params:
            decl["parameters_json_schema"] = params
        function_declarations.append(decl)

    if not function_declarations:
        return None

    return [{"function_declarations": function_declarations}]


def generate_completion_id() -> str:
    """Generate a unique completion ID in OpenAI format."""
    return f"chatcmpl-{uuid.uuid4()}"


def _extract_candidate_parts(candidate: Any) -> tuple[list[str], list[OpenAIChatCompletionToolCall]]:
    """Extract text segments and tool calls from a Gemini candidate's parts."""
    content_obj = getattr(candidate, "content", None)
    parts = getattr(content_obj, "parts", None) or []

    text_parts: list[str] = []
    tool_calls: list[OpenAIChatCompletionToolCall] = []

    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text is not None:
            text_parts.append(part_text)
            continue

        fc = getattr(part, "function_call", None)
        if fc is not None:
            tool_calls.append(
                OpenAIChatCompletionToolCall(
                    index=len(tool_calls),
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=OpenAIChatCompletionToolCallFunction(
                        name=getattr(fc, "name", "") or "",
                        arguments=json.dumps(getattr(fc, "args", {}) or {}),
                    ),
                )
            )

    return text_parts, tool_calls


def _iter_candidate_outputs(
    response_or_chunk: Any,
) -> list[tuple[int, str | None, list[OpenAIChatCompletionToolCall], Any]]:
    outputs: list[tuple[int, str | None, list[OpenAIChatCompletionToolCall], Any]] = []
    for i, candidate in enumerate(getattr(response_or_chunk, "candidates", None) or []):
        text_parts, tool_calls = _extract_candidate_parts(candidate)
        outputs.append(
            (
                i,
                "".join(text_parts) if text_parts else None,
                tool_calls,
                getattr(candidate, "finish_reason", None),
            )
        )
    return outputs


def _resolve_finish_reason_common(
    finish_reason_val: Any,
    has_tool_calls: bool,
    *,
    allow_none: bool,
) -> OpenAIFinishReason | None:
    if has_tool_calls:
        return "tool_calls"
    if finish_reason_val is None:
        return None if allow_none else "stop"
    return convert_finish_reason(str(finish_reason_val))


def _resolve_finish_reason(
    finish_reason_val: Any,
    has_tool_calls: bool,
) -> OpenAIFinishReason:
    """Determine the OpenAI finish reason for a candidate."""
    finish_reason = _resolve_finish_reason_common(finish_reason_val, has_tool_calls, allow_none=False)
    return "stop" if finish_reason is None else finish_reason


def convert_gemini_response_to_openai(
    response: genai_types.GenerateContentResponse,
    model: str,
) -> OpenAIChatCompletion:
    """Map a google-genai ``GenerateContentResponse`` to ``OpenAIChatCompletion``."""
    completion_id = generate_completion_id()
    created = int(time.time())

    choices: list[OpenAIChoice] = []

    for i, content_text, tool_calls, finish_reason_raw in _iter_candidate_outputs(response):
        finish_reason = _resolve_finish_reason(finish_reason_raw, bool(tool_calls))

        choices.append(
            OpenAIChoice(
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content=content_text,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason,
                index=i,
            )
        )

    if not choices:
        choices.append(
            OpenAIChoice(
                message=OpenAIChatCompletionResponseMessage(role="assistant", content=None),
                finish_reason="content_filter",
                index=0,
            )
        )

    return OpenAIChatCompletion(
        id=completion_id,
        choices=choices,
        created=created,
        model=model,
        usage=_extract_usage(response),
    )


def _extract_usage(
    response: genai_types.GenerateContentResponse,
) -> OpenAIChatCompletionUsage | None:
    """Extract token usage from a Gemini response."""
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is None:
        return None

    prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage_meta, "total_token_count", 0) or 0

    return OpenAIChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _resolve_stream_finish_reason(
    finish_reason_val: Any,
    has_tool_calls: bool,
) -> OpenAIFinishReason | None:
    """Determine the OpenAI finish reason for a streaming chunk candidate.

    Unlike the non-streaming variant, returns ``None`` when the candidate has
    no finish reason yet (mid-stream).
    """
    return _resolve_finish_reason_common(finish_reason_val, has_tool_calls, allow_none=True)


def convert_gemini_stream_chunk_to_openai(
    chunk: genai_types.GenerateContentResponse,
    model: str,
    completion_id: str,
    is_first_chunk: bool,
) -> OpenAIChatCompletionChunk:
    """Map a Gemini streaming chunk to ``OpenAIChatCompletionChunk``."""
    created = int(time.time())
    role = "assistant" if is_first_chunk else None

    choices: list[OpenAIChunkChoice] = []

    for i, content_text, tool_calls, finish_reason_raw in _iter_candidate_outputs(chunk):
        finish_reason = _resolve_stream_finish_reason(finish_reason_raw, bool(tool_calls))

        choices.append(
            OpenAIChunkChoice(
                delta=OpenAIChoiceDelta(
                    role=role,
                    content=content_text,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason,
                index=i,
            )
        )

    if not choices:
        choices.append(
            OpenAIChunkChoice(
                delta=OpenAIChoiceDelta(role=role, content=None),
                finish_reason=None,
                index=0,
            )
        )

    return OpenAIChatCompletionChunk(
        id=completion_id,
        choices=choices,
        created=created,
        model=model,
        usage=_extract_usage(chunk),
    )
