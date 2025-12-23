# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import (
    Any,
)

from openai.types.chat import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    StopReason,
    ToolCall,
    ToolDefinition,
)

logger = get_logger(name=__name__, category="providers::utils")


class OpenAICompatCompletionChoiceDelta(BaseModel):
    content: str


class OpenAICompatLogprobs(BaseModel):
    text_offset: list[int] | None = None

    token_logprobs: list[float] | None = None

    tokens: list[str] | None = None

    top_logprobs: list[dict[str, float]] | None = None


class OpenAICompatCompletionChoice(BaseModel):
    finish_reason: str | None = None
    text: str | None = None
    delta: OpenAICompatCompletionChoiceDelta | None = None
    logprobs: OpenAICompatLogprobs | None = None


class OpenAICompatCompletionResponse(BaseModel):
    choices: list[OpenAICompatCompletionChoice]


def text_from_choice(choice) -> str:
    if hasattr(choice, "delta") and choice.delta:
        return choice.delta.content  # type: ignore[no-any-return]  # external OpenAI types lack precise annotations

    if hasattr(choice, "message"):
        return choice.message.content  # type: ignore[no-any-return]  # external OpenAI types lack precise annotations

    return choice.text  # type: ignore[no-any-return]  # external OpenAI types lack precise annotations


def get_stop_reason(finish_reason: str) -> StopReason:
    if finish_reason in ["stop", "eos"]:
        return StopReason.end_of_turn
    elif finish_reason == "eom":
        return StopReason.end_of_message
    elif finish_reason == "length":
        return StopReason.out_of_tokens

    return StopReason.out_of_tokens


class UnparseableToolCall(BaseModel):
    """
    A ToolCall with arguments that are not valid JSON.
    Mirrors the ToolCall schema, but with arguments as a string.
    """

    call_id: str = ""
    tool_name: str = ""
    arguments: str = ""


def convert_tool_call(
    tool_call: ChatCompletionMessageToolCall,
) -> ToolCall | UnparseableToolCall:
    """
    Convert a ChatCompletionMessageToolCall tool call to either a
    ToolCall or UnparseableToolCall. Returns an UnparseableToolCall
    if the tool call is not valid ToolCall.
    """
    try:
        valid_tool_call = ToolCall(
            call_id=tool_call.id,
            tool_name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
    except Exception:
        return UnparseableToolCall(
            call_id=tool_call.id or "",
            tool_name=tool_call.function.name or "",
            arguments=tool_call.function.arguments or "",
        )

    return valid_tool_call


PYTHON_TYPE_TO_LITELLM_TYPE = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
}


def to_openai_param_type(param_type: str) -> dict:
    """
    Convert Python type hints to OpenAI parameter type format.

    Examples:
        'str' -> {'type': 'string'}
        'int' -> {'type': 'integer'}
        'list[str]' -> {'type': 'array', 'items': {'type': 'string'}}
        'list[int]' -> {'type': 'array', 'items': {'type': 'integer'}}
    """
    # Handle basic types first
    basic_types = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }

    if param_type in basic_types:
        return {"type": basic_types[param_type]}

    # Handle list/array types
    if param_type.startswith("list[") and param_type.endswith("]"):
        inner_type = param_type[5:-1]
        if inner_type in basic_types:
            return {
                "type": "array",
                "items": {"type": basic_types.get(inner_type, inner_type)},
            }

    return {"type": param_type}


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


def prepare_openai_embeddings_params(
    model: str,
    input: str | list[str],
    encoding_format: str | None = "float",
    dimensions: int | None = None,
    user: str | None = None,
):
    if model is None:
        raise ValueError("Model must be provided for embeddings")

    input_list = [input] if isinstance(input, str) else input

    params: dict[str, Any] = {
        "model": model,
        "input": input_list,
    }

    if encoding_format is not None:
        params["encoding_format"] = encoding_format
    if dimensions is not None:
        params["dimensions"] = dimensions
    if user is not None:
        params["user"] = user

    return params


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
