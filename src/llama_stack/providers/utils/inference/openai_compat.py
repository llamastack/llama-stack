# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import Iterable
from typing import (
    Any,
)

from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)

try:
    from openai.types.chat import (
        ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
    )
except ImportError:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall as OpenAIChatCompletionMessageFunctionToolCall,
    )
from openai.types.chat import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from llama_stack.apis.common.content_types import (
    URL,
    ImageContentItem,
    TextContentItem,
    _URLOrData,
)
from llama_stack.apis.inference import (
    GreedySamplingStrategy,
    JsonSchemaResponseFormat,
    OpenAIResponseFormatParam,
    SamplingParams,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
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


def get_sampling_strategy_options(params: SamplingParams) -> dict:
    options = {}
    if isinstance(params.strategy, GreedySamplingStrategy):
        options["temperature"] = 0.0
    elif isinstance(params.strategy, TopPSamplingStrategy):
        if params.strategy.temperature is not None:
            options["temperature"] = params.strategy.temperature
        if params.strategy.top_p is not None:
            options["top_p"] = params.strategy.top_p
    elif isinstance(params.strategy, TopKSamplingStrategy):
        options["top_k"] = params.strategy.top_k
    else:
        raise ValueError(f"Unsupported sampling strategy: {params.strategy}")

    return options


def get_sampling_options(params: SamplingParams | None) -> dict:
    if not params:
        return {}

    options = {}
    if params:
        options.update(get_sampling_strategy_options(params))
        if params.max_tokens:
            options["max_tokens"] = params.max_tokens

        if params.repetition_penalty is not None and params.repetition_penalty != 1.0:
            options["repeat_penalty"] = params.repetition_penalty

        if params.stop is not None:
            options["stop"] = params.stop

    return options


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


def _convert_stop_reason_to_openai_finish_reason(stop_reason: StopReason) -> str:
    """
    Convert a StopReason to an OpenAI chat completion finish_reason.
    """
    return {
        StopReason.end_of_turn: "stop",
        StopReason.end_of_message: "tool_calls",
        StopReason.out_of_tokens: "length",
    }.get(stop_reason, "stop")


def _convert_openai_finish_reason(finish_reason: str) -> StopReason:
    """
    Convert an OpenAI chat completion finish_reason to a StopReason.

    finish_reason: Literal["stop", "length", "tool_calls", ...]
        - stop: model hit a natural stop point or a provided stop sequence
        - length: maximum number of tokens specified in the request was reached
        - tool_calls: model called a tool

    ->

    class StopReason(Enum):
        end_of_turn = "end_of_turn"
        end_of_message = "end_of_message"
        out_of_tokens = "out_of_tokens"
    """

    # TODO(mf): are end_of_turn and end_of_message semantics correct?
    return {
        "stop": StopReason.end_of_turn,
        "length": StopReason.out_of_tokens,
        "tool_calls": StopReason.end_of_message,
    }.get(finish_reason, StopReason.end_of_turn)


def _convert_openai_request_tools(tools: list[dict[str, Any]] | None = None) -> list[ToolDefinition]:
    lls_tools: list[ToolDefinition] = []
    if not tools:
        return lls_tools

    for tool in tools:
        tool_fn = tool.get("function", {})
        tool_name = tool_fn.get("name", None)
        tool_desc = tool_fn.get("description", None)
        tool_params = tool_fn.get("parameters", None)

        lls_tool = ToolDefinition(
            tool_name=tool_name,
            description=tool_desc,
            input_schema=tool_params,  # Pass through entire JSON Schema
        )
        lls_tools.append(lls_tool)
    return lls_tools


def _convert_openai_request_response_format(
    response_format: OpenAIResponseFormatParam | None = None,
):
    if not response_format:
        return None
    # response_format can be a dict or a pydantic model
    response_format_dict = dict(response_format)  # type: ignore[arg-type]  # OpenAIResponseFormatParam union needs dict conversion
    if response_format_dict.get("type", "") == "json_schema":
        return JsonSchemaResponseFormat(
            type="json_schema",  # type: ignore[arg-type]  # Literal["json_schema"] incompatible with expected type
            json_schema=response_format_dict.get("json_schema", {}).get("schema", ""),
        )
    return None


def _convert_openai_tool_calls(
    tool_calls: list[OpenAIChatCompletionMessageFunctionToolCall],
) -> list[ToolCall]:
    """
    Convert an OpenAI ChatCompletionMessageToolCall list into a list of ToolCall.

    OpenAI ChatCompletionMessageToolCall:
        id: str
        function: Function
        type: Literal["function"]

    OpenAI Function:
        arguments: str
        name: str

    ->

    ToolCall:
        call_id: str
        tool_name: str
        arguments: Dict[str, ...]
    """
    if not tool_calls:
        return []  # CompletionMessage tool_calls is not optional

    return [
        ToolCall(
            call_id=call.id,
            tool_name=call.function.name,
            arguments=call.function.arguments,
        )
        for call in tool_calls
    ]


def _convert_openai_sampling_params(
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> SamplingParams:
    sampling_params = SamplingParams()

    if max_tokens:
        sampling_params.max_tokens = max_tokens

    # Map an explicit temperature of 0 to greedy sampling
    if temperature == 0:
        sampling_params.strategy = GreedySamplingStrategy()
    else:
        # OpenAI defaults to 1.0 for temperature and top_p if unset
        if temperature is None:
            temperature = 1.0
        if top_p is None:
            top_p = 1.0
        sampling_params.strategy = TopPSamplingStrategy(temperature=temperature, top_p=top_p)  # type: ignore[assignment]  # SamplingParams.strategy union accepts this type

    return sampling_params


def openai_content_to_content(content: str | Iterable[OpenAIChatCompletionContentPartParam] | None):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return [openai_content_to_content(c) for c in content]
    elif hasattr(content, "type"):
        if content.type == "text":
            return TextContentItem(type="text", text=content.text)  # type: ignore[attr-defined]  # Iterable narrowed by hasattr check but mypy doesn't track
        elif content.type == "image_url":
            return ImageContentItem(type="image", image=_URLOrData(url=URL(uri=content.image_url.url)))  # type: ignore[attr-defined]  # Iterable narrowed by hasattr check but mypy doesn't track
        else:
            raise ValueError(f"Unknown content type: {content.type}")
    else:
        raise ValueError(f"Unknown content type: {content}")


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
