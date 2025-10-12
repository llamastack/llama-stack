# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable
from typing import (
    Any,
)

from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam as OpenAIChatCompletionAssistantMessage,
)
from openai.types.chat import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from openai.types.chat import (
    ChatCompletionContentPartImageParam as OpenAIChatCompletionContentPartImageParam,
)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionContentPartTextParam as OpenAIChatCompletionContentPartTextParam,
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
    ChatCompletionMessageParam as OpenAIChatCompletionMessage,
)
from openai.types.chat import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam as OpenAIChatCompletionSystemMessage,
)
from openai.types.chat import (
    ChatCompletionToolMessageParam as OpenAIChatCompletionToolMessage,
)
from openai.types.chat import (
    ChatCompletionUserMessageParam as OpenAIChatCompletionUserMessage,
)
from openai.types.chat.chat_completion import (
    Choice as OpenAIChoice,
)
from openai.types.chat.chat_completion import (
    ChoiceLogprobs as OpenAIChoiceLogprobs,  # same as chat_completion_chunk ChoiceLogprobs
)
from openai.types.chat.chat_completion_chunk import (
    Choice as OpenAIChatCompletionChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta as OpenAIChoiceDelta,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ImageURL as OpenAIImageURL,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenAIFunction,
)
from pydantic import BaseModel

from llama_stack.apis.common.content_types import (
    URL,
    ImageContentItem,
    InterleavedContent,
    TextContentItem,
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
    _URLOrData,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionResponse,
    CompletionResponseStreamChunk,
    GreedySamplingStrategy,
    JsonSchemaResponseFormat,
    Message,
    OpenAIChatCompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    SamplingParams,
    SystemMessage,
    TokenLogProbs,
    ToolChoice,
    ToolConfig,
    ToolResponseMessage,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
    UserMessage,
)
from llama_stack.apis.inference import (
    OpenAIChoice as OpenAIChatCompletionChoice,
)
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    StopReason,
    ToolCall,
    ToolDefinition,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    convert_image_content_to_url,
    decode_assistant_message,
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
        options["temperature"] = params.strategy.temperature
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
        return choice.delta.content

    if hasattr(choice, "message"):
        return choice.message.content

    return choice.text


def get_stop_reason(finish_reason: str) -> StopReason:
    if finish_reason in ["stop", "eos"]:
        return StopReason.end_of_turn
    elif finish_reason == "eom":
        return StopReason.end_of_message
    elif finish_reason == "length":
        return StopReason.out_of_tokens

    return StopReason.out_of_tokens


def convert_openai_completion_logprobs(
    logprobs: OpenAICompatLogprobs | None,
) -> list[TokenLogProbs] | None:
    if not logprobs:
        return None
    if hasattr(logprobs, "top_logprobs"):
        return [TokenLogProbs(logprobs_by_token=x) for x in logprobs.top_logprobs]

    # Together supports logprobs with top_k=1 only. This means for each token position,
    # they return only the logprobs for the selected token (vs. the top n most likely tokens).
    # Here we construct the response by matching the selected token with the logprobs.
    if logprobs.tokens and logprobs.token_logprobs:
        return [
            TokenLogProbs(logprobs_by_token={token: token_lp})
            for token, token_lp in zip(logprobs.tokens, logprobs.token_logprobs, strict=False)
        ]
    return None


def convert_openai_completion_logprobs_stream(text: str, logprobs: float | OpenAICompatLogprobs | None):
    if logprobs is None:
        return None
    if isinstance(logprobs, float):
        # Adapt response from Together CompletionChoicesChunk
        return [TokenLogProbs(logprobs_by_token={text: logprobs})]
    if hasattr(logprobs, "top_logprobs"):
        return [TokenLogProbs(logprobs_by_token=x) for x in logprobs.top_logprobs]
    return None


def process_completion_response(
    response: OpenAICompatCompletionResponse,
) -> CompletionResponse:
    choice = response.choices[0]
    # drop suffix <eot_id> if present and return stop reason as end of turn
    if choice.text.endswith("<|eot_id|>"):
        return CompletionResponse(
            stop_reason=StopReason.end_of_turn,
            content=choice.text[: -len("<|eot_id|>")],
            logprobs=convert_openai_completion_logprobs(choice.logprobs),
        )
    # drop suffix <eom_id> if present and return stop reason as end of message
    if choice.text.endswith("<|eom_id|>"):
        return CompletionResponse(
            stop_reason=StopReason.end_of_message,
            content=choice.text[: -len("<|eom_id|>")],
            logprobs=convert_openai_completion_logprobs(choice.logprobs),
        )
    return CompletionResponse(
        stop_reason=get_stop_reason(choice.finish_reason),
        content=choice.text,
        logprobs=convert_openai_completion_logprobs(choice.logprobs),
    )


def process_chat_completion_response(
    response: OpenAICompatCompletionResponse,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        if not choice.message or not choice.message.tool_calls:
            raise ValueError("Tool calls are not present in the response")

        tool_calls = [convert_tool_call(tool_call) for tool_call in choice.message.tool_calls]
        if any(isinstance(tool_call, UnparseableToolCall) for tool_call in tool_calls):
            # If we couldn't parse a tool call, jsonify the tool calls and return them
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    stop_reason=StopReason.end_of_turn,
                    content=json.dumps(tool_calls, default=lambda x: x.model_dump()),
                ),
                logprobs=None,
            )
        else:
            # Otherwise, return tool calls as normal
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    tool_calls=tool_calls,
                    stop_reason=StopReason.end_of_turn,
                    # Content is not optional
                    content="",
                ),
                logprobs=None,
            )

    # TODO: This does not work well with tool calls for vLLM remote provider
    #   Ref: https://github.com/meta-llama/llama-stack/issues/1058
    raw_message = decode_assistant_message(text_from_choice(choice), get_stop_reason(choice.finish_reason))

    # NOTE: If we do not set tools in chat-completion request, we should not
    # expect the ToolCall in the response. Instead, we should return the raw
    # response from the model.
    if raw_message.tool_calls:
        if not request.tools:
            raw_message.tool_calls = []
            raw_message.content = text_from_choice(choice)
        else:
            # only return tool_calls if provided in the request
            new_tool_calls = []
            request_tools = {t.tool_name: t for t in request.tools}
            for t in raw_message.tool_calls:
                if t.tool_name in request_tools:
                    new_tool_calls.append(t)
                else:
                    logger.warning(f"Tool {t.tool_name} not found in request tools")

            if len(new_tool_calls) < len(raw_message.tool_calls):
                raw_message.tool_calls = new_tool_calls
                raw_message.content = text_from_choice(choice)

    return ChatCompletionResponse(
        completion_message=CompletionMessage(
            content=raw_message.content,
            stop_reason=raw_message.stop_reason,
            tool_calls=raw_message.tool_calls,
        ),
        logprobs=None,
    )


async def process_completion_stream_response(
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None],
) -> AsyncGenerator[CompletionResponseStreamChunk, None]:
    stop_reason = None

    async for chunk in stream:
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason

        text = text_from_choice(choice)
        if text == "<|eot_id|>":
            stop_reason = StopReason.end_of_turn
            text = ""
            continue
        elif text == "<|eom_id|>":
            stop_reason = StopReason.end_of_message
            text = ""
            continue
        yield CompletionResponseStreamChunk(
            delta=text,
            stop_reason=stop_reason,
            logprobs=convert_openai_completion_logprobs_stream(text, choice.logprobs),
        )
        if finish_reason:
            if finish_reason in ["stop", "eos", "eos_token"]:
                stop_reason = StopReason.end_of_turn
            elif finish_reason == "length":
                stop_reason = StopReason.out_of_tokens
            break

    yield CompletionResponseStreamChunk(
        delta="",
        stop_reason=stop_reason,
    )


async def process_chat_completion_stream_response(
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None],
    request: ChatCompletionRequest,
) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.start,
            delta=TextDelta(text=""),
        )
    )

    buffer = ""
    ipython = False
    stop_reason = None

    async for chunk in stream:
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason:
            if stop_reason is None and finish_reason in ["stop", "eos", "eos_token"]:
                stop_reason = StopReason.end_of_turn
            elif stop_reason is None and finish_reason == "length":
                stop_reason = StopReason.out_of_tokens
            break

        text = text_from_choice(choice)
        if not text:
            # Sometimes you get empty chunks from providers
            continue

        # check if its a tool call ( aka starts with <|python_tag|> )
        if not ipython and text.startswith("<|python_tag|>"):
            ipython = True
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        tool_call="",
                        parse_status=ToolCallParseStatus.started,
                    ),
                )
            )
            buffer += text
            continue

        if text == "<|eot_id|>":
            stop_reason = StopReason.end_of_turn
            text = ""
            continue
        elif text == "<|eom_id|>":
            stop_reason = StopReason.end_of_message
            text = ""
            continue

        if ipython:
            buffer += text
            delta = ToolCallDelta(
                tool_call=text,
                parse_status=ToolCallParseStatus.in_progress,
            )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=delta,
                    stop_reason=stop_reason,
                )
            )
        else:
            buffer += text
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=TextDelta(text=text),
                    stop_reason=stop_reason,
                )
            )

    # parse tool calls and report errors
    message = decode_assistant_message(buffer, stop_reason)

    parsed_tool_calls = len(message.tool_calls) > 0
    if ipython and not parsed_tool_calls:
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.progress,
                delta=ToolCallDelta(
                    tool_call="",
                    parse_status=ToolCallParseStatus.failed,
                ),
                stop_reason=stop_reason,
            )
        )

    request_tools = {t.tool_name: t for t in request.tools}
    for tool_call in message.tool_calls:
        if tool_call.tool_name in request_tools:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        tool_call=tool_call,
                        parse_status=ToolCallParseStatus.succeeded,
                    ),
                    stop_reason=stop_reason,
                )
            )
        else:
            logger.warning(f"Tool {tool_call.tool_name} not found in request tools")
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        # Parsing tool call failed due to tool call not being found in request tools,
                        # We still add the raw message text inside tool_call for responding back to the user
                        tool_call=buffer,
                        parse_status=ToolCallParseStatus.failed,
                    ),
                    stop_reason=stop_reason,
                )
            )

    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.complete,
            delta=TextDelta(text=""),
            stop_reason=stop_reason,
        )
    )


async def convert_message_to_openai_dict(message: Message, download: bool = False) -> dict:
    async def _convert_content(content) -> dict:
        if isinstance(content, ImageContentItem):
            return {
                "type": "image_url",
                "image_url": {
                    "url": await convert_image_content_to_url(content, download=download),
                },
            }
        else:
            text = content.text if isinstance(content, TextContentItem) else content
            assert isinstance(text, str)
            return {"type": "text", "text": text}

    if isinstance(message.content, list):
        content = [await _convert_content(c) for c in message.content]
    else:
        content = [await _convert_content(message.content)]

    result = {
        "role": message.role,
        "content": content,
    }

    if hasattr(message, "tool_calls") and message.tool_calls:
        result["tool_calls"] = []
        for tc in message.tool_calls:
            # The tool.tool_name can be a str or a BuiltinTool enum. If
            # it's the latter, convert to a string.
            tool_name = tc.tool_name
            if isinstance(tool_name, BuiltinTool):
                tool_name = tool_name.value

            result["tool_calls"].append(
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tc.arguments,
                    },
                }
            )
    return result


class UnparseableToolCall(BaseModel):
    """
    A ToolCall with arguments that are not valid JSON.
    Mirrors the ToolCall schema, but with arguments as a string.
    """

    call_id: str = ""
    tool_name: str = ""
    arguments: str = ""


async def convert_message_to_openai_dict_new(
    message: Message | dict,
    download_images: bool = False,
) -> OpenAIChatCompletionMessage:
    """
    Convert a Message to an OpenAI API-compatible dictionary.
    """
    # users can supply a dict instead of a Message object, we'll
    # convert it to a Message object and proceed with some type safety.
    if isinstance(message, dict):
        if "role" not in message:
            raise ValueError("role is required in message")
        if message["role"] == "user":
            message = UserMessage(**message)
        elif message["role"] == "assistant":
            message = CompletionMessage(**message)
        elif message["role"] == "tool":
            message = ToolResponseMessage(**message)
        elif message["role"] == "system":
            message = SystemMessage(**message)
        else:
            raise ValueError(f"Unsupported message role: {message['role']}")

    # Map Llama Stack spec to OpenAI spec -
    #  str -> str
    #  {"type": "text", "text": ...} -> {"type": "text", "text": ...}
    #  {"type": "image", "image": {"url": {"uri": ...}}} -> {"type": "image_url", "image_url": {"url": ...}}
    #  {"type": "image", "image": {"data": ...}} -> {"type": "image_url", "image_url": {"url": "data:image/?;base64,..."}}
    #  List[...] -> List[...]
    async def _convert_message_content(
        content: InterleavedContent,
    ) -> str | Iterable[OpenAIChatCompletionContentPartParam]:
        async def impl(
            content_: InterleavedContent,
        ) -> str | OpenAIChatCompletionContentPartParam | list[OpenAIChatCompletionContentPartParam]:
            # Llama Stack and OpenAI spec match for str and text input
            if isinstance(content_, str):
                return content_
            elif isinstance(content_, TextContentItem):
                return OpenAIChatCompletionContentPartTextParam(
                    type="text",
                    text=content_.text,
                )
            elif isinstance(content_, ImageContentItem):
                return OpenAIChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=OpenAIImageURL(
                        url=await convert_image_content_to_url(content_, download=download_images)
                    ),
                )
            elif isinstance(content_, list):
                return [await impl(item) for item in content_]
            else:
                raise ValueError(f"Unsupported content type: {type(content_)}")

        ret = await impl(content)

        # OpenAI*Message expects a str or list
        if isinstance(ret, str) or isinstance(ret, list):
            return ret
        else:
            return [ret]

    out: OpenAIChatCompletionMessage = None
    if isinstance(message, UserMessage):
        out = OpenAIChatCompletionUserMessage(
            role="user",
            content=await _convert_message_content(message.content),
        )
    elif isinstance(message, CompletionMessage):
        tool_calls = [
            OpenAIChatCompletionMessageFunctionToolCall(
                id=tool.call_id,
                function=OpenAIFunction(
                    name=(tool.tool_name if not isinstance(tool.tool_name, BuiltinTool) else tool.tool_name.value),
                    arguments=tool.arguments,  # Already a JSON string, don't double-encode
                ),
                type="function",
            )
            for tool in message.tool_calls
        ]
        params = {}
        if tool_calls:
            params["tool_calls"] = tool_calls
        out = OpenAIChatCompletionAssistantMessage(
            role="assistant",
            content=await _convert_message_content(message.content),
            **params,
        )
    elif isinstance(message, ToolResponseMessage):
        out = OpenAIChatCompletionToolMessage(
            role="tool",
            tool_call_id=message.call_id,
            content=await _convert_message_content(message.content),
        )
    elif isinstance(message, SystemMessage):
        out = OpenAIChatCompletionSystemMessage(
            role="system",
            content=await _convert_message_content(message.content),
        )
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

    return out


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
        function["name"] = tool.tool_name.value
    else:
        function["name"] = tool.tool_name

    if tool.description:
        function["description"] = tool.description

    if tool.input_schema:
        # Pass through the entire JSON Schema as-is
        function["parameters"] = tool.input_schema

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


def _convert_openai_request_tool_config(tool_choice: str | dict[str, Any] | None = None) -> ToolConfig:
    tool_config = ToolConfig()
    if tool_choice:
        try:
            tool_choice = ToolChoice(tool_choice)
        except ValueError:
            pass
        tool_config.tool_choice = tool_choice
    return tool_config


def _convert_openai_request_tools(tools: list[dict[str, Any]] | None = None) -> list[ToolDefinition]:
    lls_tools = []
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
    response_format: OpenAIResponseFormatParam = None,
):
    if not response_format:
        return None
    # response_format can be a dict or a pydantic model
    response_format = dict(response_format)
    if response_format.get("type", "") == "json_schema":
        return JsonSchemaResponseFormat(
            type="json_schema",
            json_schema=response_format.get("json_schema", {}).get("schema", ""),
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


def _convert_openai_logprobs(
    logprobs: OpenAIChoiceLogprobs,
) -> list[TokenLogProbs] | None:
    """
    Convert an OpenAI ChoiceLogprobs into a list of TokenLogProbs.

    OpenAI ChoiceLogprobs:
        content: Optional[List[ChatCompletionTokenLogprob]]

    OpenAI ChatCompletionTokenLogprob:
        token: str
        logprob: float
        top_logprobs: List[TopLogprob]

    OpenAI TopLogprob:
        token: str
        logprob: float

    ->

    TokenLogProbs:
        logprobs_by_token: Dict[str, float]
         - token, logprob

    """
    if not logprobs or not logprobs.content:
        return None

    return [
        TokenLogProbs(logprobs_by_token={logprobs.token: logprobs.logprob for logprobs in content.top_logprobs})
        for content in logprobs.content
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
        strategy = GreedySamplingStrategy()
    else:
        # OpenAI defaults to 1.0 for temperature and top_p if unset
        if temperature is None:
            temperature = 1.0
        if top_p is None:
            top_p = 1.0
        strategy = TopPSamplingStrategy(temperature=temperature, top_p=top_p)

    sampling_params.strategy = strategy
    return sampling_params


def openai_messages_to_messages(
    messages: list[OpenAIMessageParam],
) -> list[Message]:
    """
    Convert a list of OpenAIChatCompletionMessage into a list of Message.
    """
    converted_messages = []
    for message in messages:
        if message.role == "system":
            converted_message = SystemMessage(content=openai_content_to_content(message.content))
        elif message.role == "user":
            converted_message = UserMessage(content=openai_content_to_content(message.content))
        elif message.role == "assistant":
            converted_message = CompletionMessage(
                content=openai_content_to_content(message.content),
                tool_calls=_convert_openai_tool_calls(message.tool_calls),
                stop_reason=StopReason.end_of_turn,
            )
        elif message.role == "tool":
            converted_message = ToolResponseMessage(
                role="tool",
                call_id=message.tool_call_id,
                content=openai_content_to_content(message.content),
            )
        else:
            raise ValueError(f"Unknown role {message.role}")
        converted_messages.append(converted_message)
    return converted_messages


def openai_content_to_content(content: str | Iterable[OpenAIChatCompletionContentPartParam] | None):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return [openai_content_to_content(c) for c in content]
    elif hasattr(content, "type"):
        if content.type == "text":
            return TextContentItem(type="text", text=content.text)
        elif content.type == "image_url":
            return ImageContentItem(type="image", image=_URLOrData(url=URL(uri=content.image_url.url)))
        else:
            raise ValueError(f"Unknown content type: {content.type}")
    else:
        raise ValueError(f"Unknown content type: {content}")


def convert_openai_chat_completion_choice(
    choice: OpenAIChoice,
) -> ChatCompletionResponse:
    """
    Convert an OpenAI Choice into a ChatCompletionResponse.

    OpenAI Choice:
        message: ChatCompletionMessage
        finish_reason: str
        logprobs: Optional[ChoiceLogprobs]

    OpenAI ChatCompletionMessage:
        role: Literal["assistant"]
        content: Optional[str]
        tool_calls: Optional[List[ChatCompletionMessageToolCall]]

    ->

    ChatCompletionResponse:
        completion_message: CompletionMessage
        logprobs: Optional[List[TokenLogProbs]]

    CompletionMessage:
        role: Literal["assistant"]
        content: str | ImageMedia | List[str | ImageMedia]
        stop_reason: StopReason
        tool_calls: List[ToolCall]

    class StopReason(Enum):
        end_of_turn = "end_of_turn"
        end_of_message = "end_of_message"
        out_of_tokens = "out_of_tokens"
    """
    assert hasattr(choice, "message") and choice.message, "error in server response: message not found"
    assert hasattr(choice, "finish_reason") and choice.finish_reason, (
        "error in server response: finish_reason not found"
    )

    return ChatCompletionResponse(
        completion_message=CompletionMessage(
            content=choice.message.content or "",  # CompletionMessage content is not optional
            stop_reason=_convert_openai_finish_reason(choice.finish_reason),
            tool_calls=_convert_openai_tool_calls(choice.message.tool_calls),
        ),
        logprobs=_convert_openai_logprobs(getattr(choice, "logprobs", None)),
    )


async def convert_openai_chat_completion_stream(
    stream: AsyncStream[OpenAIChatCompletionChunk],
    enable_incremental_tool_calls: bool,
) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
    """
    Convert a stream of OpenAI chat completion chunks into a stream
    of ChatCompletionResponseStreamChunk.
    """
    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.start,
            delta=TextDelta(text=""),
        )
    )
    event_type = ChatCompletionResponseEventType.progress

    stop_reason = None
    tool_call_idx_to_buffer = {}

    async for chunk in stream:
        choice = chunk.choices[0]  # assuming only one choice per chunk

        # we assume there's only one finish_reason in the stream
        stop_reason = _convert_openai_finish_reason(choice.finish_reason) or stop_reason
        logprobs = getattr(choice, "logprobs", None)

        # if there's a tool call, emit an event for each tool in the list
        # if tool call and content, emit both separately
        if choice.delta.tool_calls:
            # the call may have content and a tool call. ChatCompletionResponseEvent
            # does not support both, so we emit the content first
            if choice.delta.content:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=TextDelta(text=choice.delta.content),
                        logprobs=_convert_openai_logprobs(logprobs),
                    )
                )

            # it is possible to have parallel tool calls in stream, but
            # ChatCompletionResponseEvent only supports one per stream
            if len(choice.delta.tool_calls) > 1:
                warnings.warn(
                    "multiple tool calls found in a single delta, using the first, ignoring the rest",
                    stacklevel=2,
                )

            if not enable_incremental_tool_calls:
                for tool_call in choice.delta.tool_calls:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=event_type,
                            delta=ToolCallDelta(
                                tool_call=_convert_openai_tool_calls([tool_call])[0],
                                parse_status=ToolCallParseStatus.succeeded,
                            ),
                            logprobs=_convert_openai_logprobs(logprobs),
                        )
                    )
            else:
                for tool_call in choice.delta.tool_calls:
                    idx = tool_call.index if hasattr(tool_call, "index") else 0

                    if idx not in tool_call_idx_to_buffer:
                        tool_call_idx_to_buffer[idx] = {
                            "call_id": tool_call.id,
                            "name": None,
                            "arguments": "",
                            "content": "",
                        }

                    buffer = tool_call_idx_to_buffer[idx]

                    if tool_call.function:
                        if tool_call.function.name:
                            buffer["name"] = tool_call.function.name
                            delta = f"{buffer['name']}("
                            buffer["content"] += delta

                        if tool_call.function.arguments:
                            delta = tool_call.function.arguments
                            buffer["arguments"] += delta
                            buffer["content"] += delta

                        yield ChatCompletionResponseStreamChunk(
                            event=ChatCompletionResponseEvent(
                                event_type=event_type,
                                delta=ToolCallDelta(
                                    tool_call=delta,
                                    parse_status=ToolCallParseStatus.in_progress,
                                ),
                                logprobs=_convert_openai_logprobs(logprobs),
                            )
                        )
        elif choice.delta.content:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=_convert_openai_logprobs(logprobs),
                )
            )

    for idx, buffer in tool_call_idx_to_buffer.items():
        logger.debug(f"toolcall_buffer[{idx}]: {buffer}")
        if buffer["name"]:
            delta = ")"
            buffer["content"] += delta
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=ToolCallDelta(
                        tool_call=delta,
                        parse_status=ToolCallParseStatus.in_progress,
                    ),
                    logprobs=None,
                )
            )

            try:
                tool_call = ToolCall(
                    call_id=buffer["call_id"],
                    tool_name=buffer["name"],
                    arguments=buffer["arguments"],
                )
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=tool_call,
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                        stop_reason=stop_reason,
                    )
                )
            except json.JSONDecodeError as e:
                print(f"Failed to parse arguments: {e}")
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=buffer["content"],
                            parse_status=ToolCallParseStatus.failed,
                        ),
                        stop_reason=stop_reason,
                    )
                )

    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.complete,
            delta=TextDelta(text=""),
            stop_reason=stop_reason,
        )
    )


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


class OpenAIChatCompletionToLlamaStackMixin:
    async def openai_chat_completion(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        frequency_penalty: float | None = None,
        function_call: str | dict[str, Any] | None = None,
        functions: list[dict[str, Any]] | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: OpenAIResponseFormatParam | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        messages = openai_messages_to_messages(messages)
        response_format = _convert_openai_request_response_format(response_format)
        sampling_params = _convert_openai_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        tool_config = _convert_openai_request_tool_config(tool_choice)

        tools = _convert_openai_request_tools(tools)
        if tool_config.tool_choice == ToolChoice.none:
            tools = []

        outstanding_responses = []
        # "n" is the number of completions to generate per prompt
        n = n or 1
        for _i in range(0, n):
            response = self.chat_completion(
                model_id=model,
                messages=messages,
                sampling_params=sampling_params,
                response_format=response_format,
                stream=stream,
                tool_config=tool_config,
                tools=tools,
            )
            outstanding_responses.append(response)

        if stream:
            return OpenAIChatCompletionToLlamaStackMixin._process_stream_response(self, model, outstanding_responses)

        return await OpenAIChatCompletionToLlamaStackMixin._process_non_stream_response(
            self, model, outstanding_responses
        )

    async def _process_stream_response(
        self,
        model: str,
        outstanding_responses: list[Awaitable[AsyncIterator[ChatCompletionResponseStreamChunk]]],
    ):
        id = f"chatcmpl-{uuid.uuid4()}"
        for i, outstanding_response in enumerate(outstanding_responses):
            response = await outstanding_response
            async for chunk in response:
                event = chunk.event
                finish_reason = _convert_stop_reason_to_openai_finish_reason(event.stop_reason)

                if isinstance(event.delta, TextDelta):
                    text_delta = event.delta.text
                    delta = OpenAIChoiceDelta(content=text_delta)
                    yield OpenAIChatCompletionChunk(
                        id=id,
                        choices=[OpenAIChatCompletionChunkChoice(index=i, finish_reason=finish_reason, delta=delta)],
                        created=int(time.time()),
                        model=model,
                        object="chat.completion.chunk",
                    )
                elif isinstance(event.delta, ToolCallDelta):
                    if event.delta.parse_status == ToolCallParseStatus.succeeded:
                        tool_call = event.delta.tool_call

                        # First chunk includes full structure
                        openai_tool_call = OpenAIChoiceDeltaToolCall(
                            index=0,
                            id=tool_call.call_id,
                            function=OpenAIChoiceDeltaToolCallFunction(
                                name=tool_call.tool_name,
                                arguments="",
                            ),
                        )
                        delta = OpenAIChoiceDelta(tool_calls=[openai_tool_call])
                        yield OpenAIChatCompletionChunk(
                            id=id,
                            choices=[
                                OpenAIChatCompletionChunkChoice(index=i, finish_reason=finish_reason, delta=delta)
                            ],
                            created=int(time.time()),
                            model=model,
                            object="chat.completion.chunk",
                        )
                        # arguments
                        openai_tool_call = OpenAIChoiceDeltaToolCall(
                            index=0,
                            function=OpenAIChoiceDeltaToolCallFunction(
                                arguments=tool_call.arguments,
                            ),
                        )
                        delta = OpenAIChoiceDelta(tool_calls=[openai_tool_call])
                        yield OpenAIChatCompletionChunk(
                            id=id,
                            choices=[
                                OpenAIChatCompletionChunkChoice(index=i, finish_reason=finish_reason, delta=delta)
                            ],
                            created=int(time.time()),
                            model=model,
                            object="chat.completion.chunk",
                        )

    async def _process_non_stream_response(
        self, model: str, outstanding_responses: list[Awaitable[ChatCompletionResponse]]
    ) -> OpenAIChatCompletion:
        choices = []
        for outstanding_response in outstanding_responses:
            response = await outstanding_response
            completion_message = response.completion_message
            message = await convert_message_to_openai_dict_new(completion_message)
            finish_reason = _convert_stop_reason_to_openai_finish_reason(completion_message.stop_reason)

            choice = OpenAIChatCompletionChoice(
                index=len(choices),
                message=message,
                finish_reason=finish_reason,
            )
            choices.append(choice)

        return OpenAIChatCompletion(
            id=f"chatcmpl-{uuid.uuid4()}",
            choices=choices,
            created=int(time.time()),
            model=model,
            object="chat.completion",
        )


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
