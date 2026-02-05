# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import (
    Any,
)

from pydantic import BaseModel

from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import BuiltinTool, ToolDefinition

logger = get_logger(name=__name__, category="providers::utils")


def convert_tooldef_to_openai_tool(tool: ToolDefinition) -> dict:
    """
    Convert a ToolDefinition to an OpenAI API-compatible dictionary.

    ToolDefinition:
        tool_name: str | BuiltinTool
        description: Optional[str]
        input_schema: Optional[Dict[str, Any]]  # JSON Schema
        output_schema: Optional[Dict[str, Any]]  # JSON Schema (not used by OpenAI)

    OpenAI spec -

    {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {<JSON Schema>},
        },
    }

    NOTE: OpenAI does not support output_schema, so it is dropped here.
    """
    out = {
        "type": "function",
        "function": {},
    }
    function = out["function"]

    if isinstance(tool.tool_name, BuiltinTool):
        function["name"] = tool.tool_name.value  # type: ignore[index]  # dict value inferred as Any but mypy sees Collection[str]
    else:
        function["name"] = tool.tool_name  # type: ignore[index]  # dict value inferred as Any but mypy sees Collection[str]

    if tool.description:
        function["description"] = tool.description  # type: ignore[index]  # dict value inferred as Any but mypy sees Collection[str]

    if tool.input_schema:
        # Pass through the entire JSON Schema as-is
        function["parameters"] = tool.input_schema  # type: ignore[index]  # dict value inferred as Any but mypy sees Collection[str]

    # NOTE: OpenAI does not support output_schema, so we drop it here
    # It's stored in LlamaStack for validation and other provider usage

    return out


async def prepare_openai_completion_params(**params):
    async def _prepare_value(value: Any) -> Any:
        new_value = value
        if isinstance(value, list):
            new_value = [await _prepare_value(v) for v in value]
        elif isinstance(value, dict):
            new_value = {k: await _prepare_value(v) for k, v in value.items()}
        elif isinstance(value, BaseModel):
            new_value = value.model_dump(exclude_none=True)
        return new_value

    completion_params = {}
    for k, v in params.items():
        if v is not None:
            completion_params[k] = await _prepare_value(v)
    return completion_params


def get_stream_options_for_telemetry(
    stream_options: dict[str, Any] | None,
    is_streaming: bool,
    supports_stream_options: bool = True,
) -> dict[str, Any] | None:
    """
    Inject stream_options when streaming and telemetry is active.

    Active telemetry takes precedence over caller preference to ensure
    complete and consistent observability metrics.

    Args:
        stream_options: Existing stream options from the request
        is_streaming: Whether this is a streaming request
        supports_stream_options: Whether the provider supports stream_options parameter

    Returns:
        Updated stream_options with include_usage=True if conditions are met, otherwise original options
    """
    if not is_streaming:
        return stream_options

    if not supports_stream_options:
        return stream_options

    from opentelemetry import trace

    span = trace.get_current_span()
    if not span or not span.is_recording():
        return stream_options

    if stream_options is None:
        return {"include_usage": True}

    return {**stream_options, "include_usage": True}
