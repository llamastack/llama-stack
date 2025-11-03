# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from llama_stack.apis.common.content_types import ContentDelta, InterleavedContent
from llama_stack.apis.models import Model
from llama_stack.core.telemetry.telemetry import MetricResponseMixin
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.schema_utils import json_schema_type, register_schema

register_schema(ToolCall)
register_schema(ToolDefinition)


@json_schema_type
class GreedySamplingStrategy(BaseModel):
    """Greedy sampling strategy that selects the highest probability token at each step."""

    type: Literal["greedy"] = "greedy"


@json_schema_type
class TopPSamplingStrategy(BaseModel):
    """Top-p (nucleus) sampling strategy that samples from the smallest set of tokens with cumulative probability >= p."""

    type: Literal["top_p"] = "top_p"
    temperature: float | None = Field(..., gt=0.0)
    top_p: float | None = 0.95


@json_schema_type
class TopKSamplingStrategy(BaseModel):
    """Top-k sampling strategy that restricts sampling to the k most likely tokens."""

    type: Literal["top_k"] = "top_k"
    top_k: int = Field(..., ge=1)


SamplingStrategy = Annotated[
    GreedySamplingStrategy | TopPSamplingStrategy | TopKSamplingStrategy,
    Field(discriminator="type"),
]
register_schema(SamplingStrategy, name="SamplingStrategy")


@json_schema_type
class SamplingParams(BaseModel):
    """Sampling parameters."""

    strategy: SamplingStrategy = Field(default_factory=GreedySamplingStrategy)

    max_tokens: int | None = None
    repetition_penalty: float | None = 1.0
    stop: list[str] | None = None


class LogProbConfig(BaseModel):
    """Configuration for log probability generation."""

    top_k: int | None = 0


class QuantizationType(Enum):
    """Type of model quantization to run inference with."""

    bf16 = "bf16"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    """Configuration for 8-bit floating point quantization."""

    type: Literal["fp8_mixed"] = "fp8_mixed"


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    """Configuration for BFloat16 precision (typically no quantization)."""

    type: Literal["bf16"] = "bf16"


@json_schema_type
class Int4QuantizationConfig(BaseModel):
    """Configuration for 4-bit integer quantization."""

    type: Literal["int4_mixed"] = "int4_mixed"
    scheme: str | None = "int4_weight_int8_dynamic_activation"


QuantizationConfig = Annotated[
    Bf16QuantizationConfig | Fp8QuantizationConfig | Int4QuantizationConfig,
    Field(discriminator="type"),
]


@json_schema_type
class UserMessage(BaseModel):
    """A message from the user in a chat conversation."""

    role: Literal["user"] = "user"
    content: InterleavedContent
    context: InterleavedContent | None = None


@json_schema_type
class SystemMessage(BaseModel):
    """A system message providing instructions or context to the model."""

    role: Literal["system"] = "system"
    content: InterleavedContent


@json_schema_type
class ToolResponseMessage(BaseModel):
    """A message representing the result of a tool invocation."""

    role: Literal["tool"] = "tool"
    call_id: str
    content: InterleavedContent


@json_schema_type
class CompletionMessage(BaseModel):
    """A message containing the model's (assistant) response in a chat conversation.

    - `StopReason.end_of_turn`: The model finished generating the entire response.
    - `StopReason.end_of_message`: The model finished generating but generated a partial response -- usually, a tool call. The user may call the tool and continue the conversation with the tool's response.
    - `StopReason.out_of_tokens`: The model ran out of token budget.
    """

    role: Literal["assistant"] = "assistant"
    content: InterleavedContent
    stop_reason: StopReason
    tool_calls: list[ToolCall] | None = Field(default_factory=lambda: [])


Message = Annotated[
    UserMessage | SystemMessage | ToolResponseMessage | CompletionMessage,
    Field(discriminator="role"),
]
register_schema(Message, name="Message")


@json_schema_type
class ToolResponse(BaseModel):
    """Response from a tool invocation."""

    call_id: str
    tool_name: BuiltinTool | str
    content: InterleavedContent
    metadata: dict[str, Any] | None = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class ToolChoice(Enum):
    """Whether tool use is required or automatic. This is a hint to the model which may not be followed. It depends on the Instruction Following capabilities of the model."""

    auto = "auto"
    required = "required"
    none = "none"


@json_schema_type
class TokenLogProbs(BaseModel):
    """Log probabilities for generated tokens."""

    logprobs_by_token: dict[str, float]


class ChatCompletionResponseEventType(Enum):
    """Types of events that can occur during chat completion."""

    start = "start"
    complete = "complete"
    progress = "progress"


@json_schema_type
class ChatCompletionResponseEvent(BaseModel):
    """An event during chat completion generation."""

    event_type: ChatCompletionResponseEventType
    delta: ContentDelta
    logprobs: list[TokenLogProbs] | None = None
    stop_reason: StopReason | None = None


class ResponseFormatType(StrEnum):
    """Types of formats for structured (guided) decoding."""

    json_schema = "json_schema"
    grammar = "grammar"


@json_schema_type
class JsonSchemaResponseFormat(BaseModel):
    """Configuration for JSON schema-guided response generation."""

    type: Literal[ResponseFormatType.json_schema] = ResponseFormatType.json_schema
    json_schema: dict[str, Any]


@json_schema_type
class GrammarResponseFormat(BaseModel):
    """Configuration for grammar-guided response generation."""

    type: Literal[ResponseFormatType.grammar] = ResponseFormatType.grammar
    bnf: dict[str, Any]


ResponseFormat = Annotated[
    JsonSchemaResponseFormat | GrammarResponseFormat,
    Field(discriminator="type"),
]
register_schema(ResponseFormat, name="ResponseFormat")


# This is an internally used class
class CompletionRequest(BaseModel):
    content: InterleavedContent
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)
    response_format: ResponseFormat | None = None
    stream: bool | None = False
    logprobs: LogProbConfig | None = None


@json_schema_type
class CompletionResponse(MetricResponseMixin):
    """Response from a completion request."""

    content: str
    stop_reason: StopReason
    logprobs: list[TokenLogProbs] | None = None


@json_schema_type
class CompletionResponseStreamChunk(MetricResponseMixin):
    """A chunk of a streamed completion response."""

    delta: str
    stop_reason: StopReason | None = None
    logprobs: list[TokenLogProbs] | None = None


class SystemMessageBehavior(Enum):
    """Config for how to override the default system prompt.

    https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-
    '{{function_definitions}}' to indicate where the function definitions should be inserted.
    """

    append = "append"
    replace = "replace"


@json_schema_type
class ToolConfig(BaseModel):
    """Configuration for tool use.

    - `ToolPromptFormat.json`: The tool calls are formatted as a JSON object.
    - `ToolPromptFormat.function_tag`: The tool calls are enclosed in a <function=function_name> tag.
    - `ToolPromptFormat.python_list`: The tool calls are output as Python syntax -- a list of function calls.
    - `SystemMessageBehavior.append`: Appends the provided system message to the default system prompt.
    - `SystemMessageBehavior.replace`: Replaces the default system prompt with the provided system message. The system message can include the string
        '{{function_definitions}}' to indicate where the function definitions should be inserted.
    """

    tool_choice: ToolChoice | str | None = Field(default=ToolChoice.auto)
    tool_prompt_format: ToolPromptFormat | None = Field(default=None)
    system_message_behavior: SystemMessageBehavior | None = Field(default=SystemMessageBehavior.append)

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.tool_choice, str):
            try:
                self.tool_choice = ToolChoice[self.tool_choice]
            except KeyError:
                pass


# This is an internally used class
@json_schema_type
class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)

    tools: list[ToolDefinition] | None = Field(default_factory=lambda: [])
    tool_config: ToolConfig | None = Field(default_factory=ToolConfig)

    response_format: ResponseFormat | None = None
    stream: bool | None = False
    logprobs: LogProbConfig | None = None


@json_schema_type
class ChatCompletionResponseStreamChunk(MetricResponseMixin):
    """A chunk of a streamed chat completion response."""

    event: ChatCompletionResponseEvent


@json_schema_type
class ChatCompletionResponse(MetricResponseMixin):
    """Response from a chat completion request."""

    completion_message: CompletionMessage
    logprobs: list[TokenLogProbs] | None = None


@json_schema_type
class EmbeddingsResponse(BaseModel):
    """Response containing generated embeddings."""

    embeddings: list[list[float]]


@json_schema_type
class RerankData(BaseModel):
    """A single rerank result from a reranking response."""

    index: int
    relevance_score: float


@json_schema_type
class RerankResponse(BaseModel):
    """Response from a reranking request."""

    data: list[RerankData]


@json_schema_type
class OpenAIChatCompletionContentPartTextParam(BaseModel):
    """Text content part for OpenAI-compatible chat completion messages."""

    type: Literal["text"] = "text"
    text: str


@json_schema_type
class OpenAIImageURL(BaseModel):
    """Image URL specification for OpenAI-compatible chat completion messages."""

    url: str
    detail: str | None = None


@json_schema_type
class OpenAIChatCompletionContentPartImageParam(BaseModel):
    """Image content part for OpenAI-compatible chat completion messages."""

    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageURL


@json_schema_type
class OpenAIFileFile(BaseModel):
    file_id: str | None = None
    filename: str | None = None


@json_schema_type
class OpenAIFile(BaseModel):
    type: Literal["file"] = "file"
    file: OpenAIFileFile


OpenAIChatCompletionContentPartParam = Annotated[
    OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam | OpenAIFile,
    Field(discriminator="type"),
]
register_schema(OpenAIChatCompletionContentPartParam, name="OpenAIChatCompletionContentPartParam")


OpenAIChatCompletionMessageContent = str | list[OpenAIChatCompletionContentPartParam]

OpenAIChatCompletionTextOnlyMessageContent = str | list[OpenAIChatCompletionContentPartTextParam]


@json_schema_type
class OpenAIUserMessageParam(BaseModel):
    """A message from the user in an OpenAI-compatible chat completion request."""

    role: Literal["user"] = "user"
    content: OpenAIChatCompletionMessageContent
    name: str | None = None


@json_schema_type
class OpenAISystemMessageParam(BaseModel):
    """A system message providing instructions or context to the model."""

    role: Literal["system"] = "system"
    content: OpenAIChatCompletionTextOnlyMessageContent
    name: str | None = None


@json_schema_type
class OpenAIChatCompletionToolCallFunction(BaseModel):
    """Function call details for OpenAI-compatible tool calls."""

    name: str | None = None
    arguments: str | None = None


@json_schema_type
class OpenAIChatCompletionToolCall(BaseModel):
    """Tool call specification for OpenAI-compatible chat completion responses."""

    index: int | None = None
    id: str | None = None
    type: Literal["function"] = "function"
    function: OpenAIChatCompletionToolCallFunction | None = None


@json_schema_type
class OpenAIAssistantMessageParam(BaseModel):
    """A message containing the model's (assistant) response in an OpenAI-compatible chat completion request."""

    role: Literal["assistant"] = "assistant"
    content: OpenAIChatCompletionTextOnlyMessageContent | None = None
    name: str | None = None
    tool_calls: list[OpenAIChatCompletionToolCall] | None = None


@json_schema_type
class OpenAIToolMessageParam(BaseModel):
    """A message representing the result of a tool invocation in an OpenAI-compatible chat completion request."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: OpenAIChatCompletionTextOnlyMessageContent


@json_schema_type
class OpenAIDeveloperMessageParam(BaseModel):
    """A message from the developer in an OpenAI-compatible chat completion request."""

    role: Literal["developer"] = "developer"
    content: OpenAIChatCompletionTextOnlyMessageContent
    name: str | None = None


OpenAIMessageParam = Annotated[
    OpenAIUserMessageParam
    | OpenAISystemMessageParam
    | OpenAIAssistantMessageParam
    | OpenAIToolMessageParam
    | OpenAIDeveloperMessageParam,
    Field(discriminator="role"),
]
register_schema(OpenAIMessageParam, name="OpenAIMessageParam")


@json_schema_type
class OpenAIResponseFormatText(BaseModel):
    """Text response format for OpenAI-compatible chat completion requests."""

    type: Literal["text"] = "text"


@json_schema_type
class OpenAIJSONSchema(TypedDict, total=False):
    """JSON schema specification for OpenAI-compatible structured response format."""

    name: str
    description: str | None
    strict: bool | None

    # Pydantic BaseModel cannot be used with a schema param, since it already
    # has one. And, we don't want to alias here because then have to handle
    # that alias when converting to OpenAI params. So, to support schema,
    # we use a TypedDict.
    schema: dict[str, Any] | None


@json_schema_type
class OpenAIResponseFormatJSONSchema(BaseModel):
    """JSON schema response format for OpenAI-compatible chat completion requests."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: OpenAIJSONSchema


@json_schema_type
class OpenAIResponseFormatJSONObject(BaseModel):
    """JSON object response format for OpenAI-compatible chat completion requests."""

    type: Literal["json_object"] = "json_object"


OpenAIResponseFormatParam = Annotated[
    OpenAIResponseFormatText | OpenAIResponseFormatJSONSchema | OpenAIResponseFormatJSONObject,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseFormatParam, name="OpenAIResponseFormatParam")


@json_schema_type
class OpenAITopLogProb(BaseModel):
    """The top log probability for a token from an OpenAI-compatible chat completion response.

    :token: The token
    :bytes: (Optional) The bytes for the token
    :logprob: The log probability of the token
    """

    token: str
    bytes: list[int] | None = None
    logprob: float


@json_schema_type
class OpenAITokenLogProb(BaseModel):
    """The log probability for a token from an OpenAI-compatible chat completion response.

    :token: The token
    :bytes: (Optional) The bytes for the token
    :logprob: The log probability of the token
    :top_logprobs: The top log probabilities for the token
    """

    token: str
    bytes: list[int] | None = None
    logprob: float
    top_logprobs: list[OpenAITopLogProb]


@json_schema_type
class OpenAIChoiceLogprobs(BaseModel):
    """The log probabilities for the tokens in the message from an OpenAI-compatible chat completion response."""

    content: list[OpenAITokenLogProb] | None = None
    refusal: list[OpenAITokenLogProb] | None = None


@json_schema_type
class OpenAIChoiceDelta(BaseModel):
    """A delta from an OpenAI-compatible chat completion streaming response."""

    content: str | None = None
    refusal: str | None = None
    role: str | None = None
    tool_calls: list[OpenAIChatCompletionToolCall] | None = None
    reasoning_content: str | None = None


@json_schema_type
class OpenAIChunkChoice(BaseModel):
    """A chunk choice from an OpenAI-compatible chat completion streaming response."""

    delta: OpenAIChoiceDelta
    finish_reason: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


@json_schema_type
class OpenAIChoice(BaseModel):
    """A choice from an OpenAI-compatible chat completion response."""

    message: OpenAIMessageParam
    finish_reason: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


class OpenAIChatCompletionUsageCompletionTokensDetails(BaseModel):
    """Token details for output tokens in OpenAI chat completion usage."""

    reasoning_tokens: int | None = None


class OpenAIChatCompletionUsagePromptTokensDetails(BaseModel):
    """Token details for prompt tokens in OpenAI chat completion usage."""

    cached_tokens: int | None = None


@json_schema_type
class OpenAIChatCompletionUsage(BaseModel):
    """Usage information for OpenAI chat completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: OpenAIChatCompletionUsagePromptTokensDetails | None = None
    completion_tokens_details: OpenAIChatCompletionUsageCompletionTokensDetails | None = None


@json_schema_type
class OpenAIChatCompletion(BaseModel):
    """Response from an OpenAI-compatible chat completion request."""

    id: str
    choices: list[OpenAIChoice]
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    usage: OpenAIChatCompletionUsage | None = None


@json_schema_type
class OpenAIChatCompletionChunk(BaseModel):
    """Chunk from a streaming response to an OpenAI-compatible chat completion request."""

    id: str
    choices: list[OpenAIChunkChoice]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    usage: OpenAIChatCompletionUsage | None = None


@json_schema_type
class OpenAICompletionLogprobs(BaseModel):
    """The log probabilities for the tokens in the message from an OpenAI-compatible completion response.

    :text_offset: (Optional) The offset of the token in the text
    :token_logprobs: (Optional) The log probabilities for the tokens
    :tokens: (Optional) The tokens
    :top_logprobs: (Optional) The top log probabilities for the tokens
    """

    text_offset: list[int] | None = None
    token_logprobs: list[float] | None = None
    tokens: list[str] | None = None
    top_logprobs: list[dict[str, float]] | None = None


@json_schema_type
class OpenAICompletionChoice(BaseModel):
    """A choice from an OpenAI-compatible completion response.

    :finish_reason: The reason the model stopped generating
    :text: The text of the choice
    :index: The index of the choice
    :logprobs: (Optional) The log probabilities for the tokens in the choice
    """

    finish_reason: str
    text: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


@json_schema_type
class OpenAICompletion(BaseModel):
    """Response from an OpenAI-compatible completion request.

    :id: The ID of the completion
    :choices: List of choices
    :created: The Unix timestamp in seconds when the completion was created
    :model: The model that was used to generate the completion
    :object: The object type, which will be "text_completion"
    """

    id: str
    choices: list[OpenAICompletionChoice]
    created: int
    model: str
    object: Literal["text_completion"] = "text_completion"


@json_schema_type
class OpenAIEmbeddingData(BaseModel):
    """A single embedding data object from an OpenAI-compatible embeddings response."""

    object: Literal["embedding"] = "embedding"
    # TODO: consider dropping str and using openai.types.embeddings.Embedding instead of OpenAIEmbeddingData
    embedding: list[float] | str
    index: int


@json_schema_type
class OpenAIEmbeddingUsage(BaseModel):
    """Usage information for an OpenAI-compatible embeddings response."""

    prompt_tokens: int
    total_tokens: int


@json_schema_type
class OpenAIEmbeddingsResponse(BaseModel):
    """Response from an OpenAI-compatible embeddings request."""

    object: Literal["list"] = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: OpenAIEmbeddingUsage


class ModelStore(Protocol):
    async def get_model(self, identifier: str) -> Model: ...


class TextTruncation(Enum):
    """Config for how to truncate text for embedding when text is longer than the model's max sequence length. Start and End semantics depend on whether the language is left-to-right or right-to-left."""

    none = "none"
    start = "start"
    end = "end"


class EmbeddingTaskType(Enum):
    """How is the embedding being used? This is only supported by asymmetric embedding models."""

    query = "query"
    document = "document"


class OpenAICompletionWithInputMessages(OpenAIChatCompletion):
    input_messages: list[OpenAIMessageParam]


@json_schema_type
class ListOpenAIChatCompletionResponse(BaseModel):
    """Response from listing OpenAI-compatible chat completions."""

    data: list[OpenAICompletionWithInputMessages]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICompletionRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible completion endpoint."""

    # Standard OpenAI completion parameters
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    best_of: int | None = None
    echo: bool | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = Field(None, ge=0, le=5)
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None
    suffix: str | None = None


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAIChatCompletionRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible chat completion endpoint."""

    # Standard OpenAI chat completion parameters
    model: str
    messages: Annotated[list[OpenAIMessageParam], Field(..., min_length=1)]
    frequency_penalty: float | None = None
    function_call: str | dict[str, Any] | None = None
    functions: list[dict[str, Any]] | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    response_format: OpenAIResponseFormatParam | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    user: str | None = None


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAIEmbeddingsRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible embeddings endpoint."""

    model: str
    input: str | list[str]
    encoding_format: str | None = "float"
    dimensions: int | None = None
    user: str | None = None
