# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from enum import Enum, StrEnum
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from fastapi import Body
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.common.responses import (
    Order,
)
from llama_stack.apis.common.tracing import telemetry_traceable
from llama_stack.apis.models import Model
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class GreedySamplingStrategy(BaseModel):
    """Greedy sampling strategy that selects the highest probability token at each step.

    :param type: Must be "greedy" to identify this sampling strategy
    """

    type: Literal["greedy"] = "greedy"


@json_schema_type
class TopPSamplingStrategy(BaseModel):
    """Top-p (nucleus) sampling strategy that samples from the smallest set of tokens with cumulative probability >= p.

    :param type: Must be "top_p" to identify this sampling strategy
    :param temperature: Controls randomness in sampling. Higher values increase randomness
    :param top_p: Cumulative probability threshold for nucleus sampling. Defaults to 0.95
    """

    type: Literal["top_p"] = "top_p"
    temperature: float | None = Field(..., gt=0.0)
    top_p: float | None = 0.95


@json_schema_type
class TopKSamplingStrategy(BaseModel):
    """Top-k sampling strategy that restricts sampling to the k most likely tokens.

    :param type: Must be "top_k" to identify this sampling strategy
    :param top_k: Number of top tokens to consider for sampling. Must be at least 1
    """

    type: Literal["top_k"] = "top_k"
    top_k: int = Field(..., ge=1)


SamplingStrategy = Annotated[
    GreedySamplingStrategy | TopPSamplingStrategy | TopKSamplingStrategy,
    Field(discriminator="type"),
]
register_schema(SamplingStrategy, name="SamplingStrategy")


@json_schema_type
class SamplingParams(BaseModel):
    """Sampling parameters.

    :param strategy: The sampling strategy.
    :param max_tokens: The maximum number of tokens that can be generated in the completion. The token count of
        your prompt plus max_tokens cannot exceed the model's context length.
    :param repetition_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens
        based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    :param stop: Up to 4 sequences where the API will stop generating further tokens.
        The returned text will not contain the stop sequence.
    """

    strategy: SamplingStrategy = Field(default_factory=GreedySamplingStrategy)

    max_tokens: int | None = None
    repetition_penalty: float | None = 1.0
    stop: list[str] | None = None


class LogProbConfig(BaseModel):
    """

    :param top_k: How many tokens (for each position) to return log probabilities for.
    """

    top_k: int | None = 0


class QuantizationType(Enum):
    """Type of model quantization to run inference with.

    :cvar bf16: BFloat16 typically this means _no_ quantization
    :cvar fp8_mixed: 8-bit floating point quantization with mixed precision
    :cvar int4_mixed: 4-bit integer quantization with mixed precision
    """

    bf16 = "bf16"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    """Configuration for 8-bit floating point quantization.

    :param type: Must be "fp8_mixed" to identify this quantization type
    """

    type: Literal["fp8_mixed"] = "fp8_mixed"


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    """Configuration for BFloat16 precision (typically no quantization).

    :param type: Must be "bf16" to identify this quantization type
    """

    type: Literal["bf16"] = "bf16"


@json_schema_type
class Int4QuantizationConfig(BaseModel):
    """Configuration for 4-bit integer quantization.

    :param type: Must be "int4" to identify this quantization type
    :param scheme: Quantization scheme to use. Defaults to "int4_weight_int8_dynamic_activation"
    """

    type: Literal["int4_mixed"] = "int4_mixed"
    scheme: str | None = "int4_weight_int8_dynamic_activation"


QuantizationConfig = Annotated[
    Bf16QuantizationConfig | Fp8QuantizationConfig | Int4QuantizationConfig,
    Field(discriminator="type"),
]


@json_schema_type
class UserMessage(BaseModel):
    """A message from the user in a chat conversation.

    :param role: Must be "user" to identify this as a user message
    :param content: The content of the message, which can include text and other media
    :param context: (Optional) This field is used internally by Llama Stack to pass RAG context. This field may be removed in the API in the future.
    """

    role: Literal["user"] = "user"
    content: InterleavedContent
    context: InterleavedContent | None = None


@json_schema_type
class SystemMessage(BaseModel):
    """A system message providing instructions or context to the model.

    :param role: Must be "system" to identify this as a system message
    :param content: The content of the "system prompt". If multiple system messages are provided, they are concatenated. The underlying Llama Stack code may also add other system messages (for example, for formatting tool definitions).
    """

    role: Literal["system"] = "system"
    content: InterleavedContent


@json_schema_type
class ToolResponseMessage(BaseModel):
    """A message representing the result of a tool invocation.

    :param role: Must be "tool" to identify this as a tool response
    :param call_id: Unique identifier for the tool call this response is for
    :param content: The response content from the tool
    """

    role: Literal["tool"] = "tool"
    call_id: str
    content: InterleavedContent


class ToolChoice(Enum):
    """Whether tool use is required or automatic. This is a hint to the model which may not be followed. It depends on the Instruction Following capabilities of the model.

    :cvar auto: The model may use tools if it determines that is appropriate.
    :cvar required: The model must use tools.
    :cvar none: The model must not use tools.
    """

    auto = "auto"
    required = "required"
    none = "none"


@json_schema_type
class TokenLogProbs(BaseModel):
    """Log probabilities for generated tokens.

    :param logprobs_by_token: Dictionary mapping tokens to their log probabilities
    """

    logprobs_by_token: dict[str, float]


class ChatCompletionResponseEventType(Enum):
    """Types of events that can occur during chat completion.

    :cvar start: Inference has started
    :cvar complete: Inference is complete and a full response is available
    :cvar progress: Inference is in progress and a partial response is available
    """

    start = "start"
    complete = "complete"
    progress = "progress"


class ResponseFormatType(StrEnum):
    """Types of formats for structured (guided) decoding.

    :cvar json_schema: Response should conform to a JSON schema. In a Python SDK, this is often a `pydantic` model.
    :cvar grammar: Response should conform to a BNF grammar
    """

    json_schema = "json_schema"
    grammar = "grammar"


@json_schema_type
class JsonSchemaResponseFormat(BaseModel):
    """Configuration for JSON schema-guided response generation.

    :param type: Must be "json_schema" to identify this format type
    :param json_schema: The JSON schema the response should conform to. In a Python SDK, this is often a `pydantic` model.
    """

    type: Literal[ResponseFormatType.json_schema] = ResponseFormatType.json_schema
    json_schema: dict[str, Any]


@json_schema_type
class GrammarResponseFormat(BaseModel):
    """Configuration for grammar-guided response generation.

    :param type: Must be "grammar" to identify this format type
    :param bnf: The BNF grammar specification the response should conform to
    """

    type: Literal[ResponseFormatType.grammar] = ResponseFormatType.grammar
    bnf: dict[str, Any]


ResponseFormat = Annotated[
    JsonSchemaResponseFormat | GrammarResponseFormat,
    Field(discriminator="type"),
]
register_schema(ResponseFormat, name="ResponseFormat")


# This is an internally used class
class CompletionRequest(BaseModel):
    model: str
    content: InterleavedContent
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)
    response_format: ResponseFormat | None = None
    stream: bool | None = False
    logprobs: LogProbConfig | None = None


class SystemMessageBehavior(Enum):
    """Config for how to override the default system prompt.

    :cvar append: Appends the provided system message to the default system prompt:
        https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-
    :cvar replace: Replaces the default system prompt with the provided system message. The system message can include the string
        '{{function_definitions}}' to indicate where the function definitions should be inserted.
    """

    append = "append"
    replace = "replace"


@json_schema_type
class EmbeddingsResponse(BaseModel):
    """Response containing generated embeddings.

    :param embeddings: List of embedding vectors, one per input content. Each embedding is a list of floats. The dimensionality of the embedding is model-specific; you can check model metadata using /models/{model_id}
    """

    embeddings: list[list[float]]


@json_schema_type
class RerankData(BaseModel):
    """A single rerank result from a reranking response.

    :param index: The original index of the document in the input list
    :param relevance_score: The relevance score from the model output. Values are inverted when applicable so that higher scores indicate greater relevance.
    """

    index: int
    relevance_score: float


@json_schema_type
class RerankResponse(BaseModel):
    """Response from a reranking request.

    :param data: List of rerank result objects, sorted by relevance score (descending)
    """

    data: list[RerankData]


@json_schema_type
class OpenAIChatCompletionContentPartTextParam(BaseModel):
    """Text content part for OpenAI-compatible chat completion messages.

    :param type: Must be "text" to identify this as text content
    :param text: The text content of the message
    """

    type: Literal["text"] = "text"
    text: str


@json_schema_type
class OpenAIImageURL(BaseModel):
    """Image URL specification for OpenAI-compatible chat completion messages.

    :param url: URL of the image to include in the message
    :param detail: (Optional) Level of detail for image processing. Can be "low", "high", or "auto"
    """

    url: str
    detail: str | None = None


@json_schema_type
class OpenAIChatCompletionContentPartImageParam(BaseModel):
    """Image content part for OpenAI-compatible chat completion messages.

    :param type: Must be "image_url" to identify this as image content
    :param image_url: Image URL specification and processing details
    """

    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageURL


@json_schema_type
class OpenAIFileFile(BaseModel):
    file_data: str | None = None
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
    """A message from the user in an OpenAI-compatible chat completion request.

    :param role: Must be "user" to identify this as a user message
    :param content: The content of the message, which can include text and other media
    :param name: (Optional) The name of the user message participant.
    """

    role: Literal["user"] = "user"
    content: OpenAIChatCompletionMessageContent
    name: str | None = None


@json_schema_type
class OpenAISystemMessageParam(BaseModel):
    """A system message providing instructions or context to the model.

    :param role: Must be "system" to identify this as a system message
    :param content: The content of the "system prompt". If multiple system messages are provided, they are concatenated. The underlying Llama Stack code may also add other system messages (for example, for formatting tool definitions).
    :param name: (Optional) The name of the system message participant.
    """

    role: Literal["system"] = "system"
    content: OpenAIChatCompletionTextOnlyMessageContent
    name: str | None = None


@json_schema_type
class OpenAIChatCompletionToolCallFunction(BaseModel):
    """Function call details for OpenAI-compatible tool calls.

    :param name: (Optional) Name of the function to call
    :param arguments: (Optional) Arguments to pass to the function as a JSON string
    """

    name: str | None = None
    arguments: str | None = None


@json_schema_type
class OpenAIChatCompletionToolCall(BaseModel):
    """Tool call specification for OpenAI-compatible chat completion responses.

    :param index: (Optional) Index of the tool call in the list
    :param id: (Optional) Unique identifier for the tool call
    :param type: Must be "function" to identify this as a function call
    :param function: (Optional) Function call details
    """

    index: int | None = None
    id: str | None = None
    type: Literal["function"] = "function"
    function: OpenAIChatCompletionToolCallFunction | None = None


@json_schema_type
class OpenAIAssistantMessageParam(BaseModel):
    """A message containing the model's (assistant) response in an OpenAI-compatible chat completion request.

    :param role: Must be "assistant" to identify this as the model's response
    :param content: The content of the model's response
    :param name: (Optional) The name of the assistant message participant.
    :param tool_calls: List of tool calls. Each tool call is an OpenAIChatCompletionToolCall object.
    """

    role: Literal["assistant"] = "assistant"
    content: OpenAIChatCompletionTextOnlyMessageContent | None = None
    name: str | None = None
    tool_calls: list[OpenAIChatCompletionToolCall] | None = None


@json_schema_type
class OpenAIToolMessageParam(BaseModel):
    """A message representing the result of a tool invocation in an OpenAI-compatible chat completion request.

    :param role: Must be "tool" to identify this as a tool response
    :param tool_call_id: Unique identifier for the tool call this response is for
    :param content: The response content from the tool
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: OpenAIChatCompletionTextOnlyMessageContent


@json_schema_type
class OpenAIDeveloperMessageParam(BaseModel):
    """A message from the developer in an OpenAI-compatible chat completion request.

    :param role: Must be "developer" to identify this as a developer message
    :param content: The content of the developer message
    :param name: (Optional) The name of the developer message participant.
    """

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
    """Text response format for OpenAI-compatible chat completion requests.

    :param type: Must be "text" to indicate plain text response format
    """

    type: Literal["text"] = "text"


@json_schema_type
class OpenAIJSONSchema(TypedDict, total=False):
    """JSON schema specification for OpenAI-compatible structured response format.

    :param name: Name of the schema
    :param description: (Optional) Description of the schema
    :param strict: (Optional) Whether to enforce strict adherence to the schema
    :param schema: (Optional) The JSON schema definition
    """

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
    """JSON schema response format for OpenAI-compatible chat completion requests.

    :param type: Must be "json_schema" to indicate structured JSON response format
    :param json_schema: The JSON schema specification for the response
    """

    type: Literal["json_schema"] = "json_schema"
    json_schema: OpenAIJSONSchema


@json_schema_type
class OpenAIResponseFormatJSONObject(BaseModel):
    """JSON object response format for OpenAI-compatible chat completion requests.

    :param type: Must be "json_object" to indicate generic JSON object response format
    """

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
    """The log probabilities for the tokens in the message from an OpenAI-compatible chat completion response.

    :param content: (Optional) The log probabilities for the tokens in the message
    :param refusal: (Optional) The log probabilities for the tokens in the message
    """

    content: list[OpenAITokenLogProb] | None = None
    refusal: list[OpenAITokenLogProb] | None = None


@json_schema_type
class OpenAIChoiceDelta(BaseModel):
    """A delta from an OpenAI-compatible chat completion streaming response.

    :param content: (Optional) The content of the delta
    :param refusal: (Optional) The refusal of the delta
    :param role: (Optional) The role of the delta
    :param tool_calls: (Optional) The tool calls of the delta
    :param reasoning_content: (Optional) The reasoning content from the model (non-standard, for o1/o3 models)
    """

    content: str | None = None
    refusal: str | None = None
    role: str | None = None
    tool_calls: list[OpenAIChatCompletionToolCall] | None = None
    reasoning_content: str | None = None


@json_schema_type
class OpenAIChunkChoice(BaseModel):
    """A chunk choice from an OpenAI-compatible chat completion streaming response.

    :param delta: The delta from the chunk
    :param finish_reason: The reason the model stopped generating
    :param index: The index of the choice
    :param logprobs: (Optional) The log probabilities for the tokens in the message
    """

    delta: OpenAIChoiceDelta
    finish_reason: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


@json_schema_type
class OpenAIChoice(BaseModel):
    """A choice from an OpenAI-compatible chat completion response.

    :param message: The message from the model
    :param finish_reason: The reason the model stopped generating
    :param index: The index of the choice
    :param logprobs: (Optional) The log probabilities for the tokens in the message
    """

    message: OpenAIMessageParam
    finish_reason: str
    index: int
    logprobs: OpenAIChoiceLogprobs | None = None


class OpenAIChatCompletionUsageCompletionTokensDetails(BaseModel):
    """Token details for output tokens in OpenAI chat completion usage.

    :param reasoning_tokens: Number of tokens used for reasoning (o1/o3 models)
    """

    reasoning_tokens: int | None = None


class OpenAIChatCompletionUsagePromptTokensDetails(BaseModel):
    """Token details for prompt tokens in OpenAI chat completion usage.

    :param cached_tokens: Number of tokens retrieved from cache
    """

    cached_tokens: int | None = None


@json_schema_type
class OpenAIChatCompletionUsage(BaseModel):
    """Usage information for OpenAI chat completion.

    :param prompt_tokens: Number of tokens in the prompt
    :param completion_tokens: Number of tokens in the completion
    :param total_tokens: Total tokens used (prompt + completion)
    :param input_tokens_details: Detailed breakdown of input token usage
    :param output_tokens_details: Detailed breakdown of output token usage
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: OpenAIChatCompletionUsagePromptTokensDetails | None = None
    completion_tokens_details: OpenAIChatCompletionUsageCompletionTokensDetails | None = None


@json_schema_type
class OpenAIChatCompletion(BaseModel):
    """Response from an OpenAI-compatible chat completion request.

    :param id: The ID of the chat completion
    :param choices: List of choices
    :param object: The object type, which will be "chat.completion"
    :param created: The Unix timestamp in seconds when the chat completion was created
    :param model: The model that was used to generate the chat completion
    :param usage: Token usage information for the completion
    """

    id: str
    choices: list[OpenAIChoice]
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    usage: OpenAIChatCompletionUsage | None = None


@json_schema_type
class OpenAIChatCompletionChunk(BaseModel):
    """Chunk from a streaming response to an OpenAI-compatible chat completion request.

    :param id: The ID of the chat completion
    :param choices: List of choices
    :param object: The object type, which will be "chat.completion.chunk"
    :param created: The Unix timestamp in seconds when the chat completion was created
    :param model: The model that was used to generate the chat completion
    :param usage: Token usage information (typically included in final chunk with stream_options)
    """

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
    """A single embedding data object from an OpenAI-compatible embeddings response.

    :param object: The object type, which will be "embedding"
    :param embedding: The embedding vector as a list of floats (when encoding_format="float") or as a base64-encoded string (when encoding_format="base64")
    :param index: The index of the embedding in the input list
    """

    object: Literal["embedding"] = "embedding"
    # TODO: consider dropping str and using openai.types.embeddings.Embedding instead of OpenAIEmbeddingData
    embedding: list[float] | str
    index: int


@json_schema_type
class OpenAIEmbeddingUsage(BaseModel):
    """Usage information for an OpenAI-compatible embeddings response.

    :param prompt_tokens: The number of tokens in the input
    :param total_tokens: The total number of tokens used
    """

    prompt_tokens: int
    total_tokens: int


@json_schema_type
class OpenAIEmbeddingsResponse(BaseModel):
    """Response from an OpenAI-compatible embeddings request.

    :param object: The object type, which will be "list"
    :param data: List of embedding data objects
    :param model: The model that was used to generate the embeddings
    :param usage: Usage information
    """

    object: Literal["list"] = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: OpenAIEmbeddingUsage


class ModelStore(Protocol):
    async def get_model(self, identifier: str) -> Model: ...


class TextTruncation(Enum):
    """Config for how to truncate text for embedding when text is longer than the model's max sequence length. Start and End semantics depend on whether the language is left-to-right or right-to-left.

    :cvar none: No truncation (default). If the text is longer than the model's max sequence length, you will get an error.
    :cvar start: Truncate from the start
    :cvar end: Truncate from the end
    """

    none = "none"
    start = "start"
    end = "end"


class EmbeddingTaskType(Enum):
    """How is the embedding being used? This is only supported by asymmetric embedding models.

    :cvar query: Used for a query for semantic search.
    :cvar document: Used at indexing time when ingesting documents.
    """

    query = "query"
    document = "document"


class OpenAICompletionWithInputMessages(OpenAIChatCompletion):
    input_messages: list[OpenAIMessageParam]


@json_schema_type
class ListOpenAIChatCompletionResponse(BaseModel):
    """Response from listing OpenAI-compatible chat completions.

    :param data: List of chat completion objects with their input messages
    :param has_more: Whether there are more completions available beyond this list
    :param first_id: ID of the first completion in this list
    :param last_id: ID of the last completion in this list
    :param object: Must be "list" to identify this as a list response
    """

    data: list[OpenAICompletionWithInputMessages]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICompletionRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible completion endpoint.

    :param model: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
    :param prompt: The prompt to generate a completion for.
    :param best_of: (Optional) The number of completions to generate.
    :param echo: (Optional) Whether to echo the prompt.
    :param frequency_penalty: (Optional) The penalty for repeated tokens.
    :param logit_bias: (Optional) The logit bias to use.
    :param logprobs: (Optional) The log probabilities to use.
    :param max_tokens: (Optional) The maximum number of tokens to generate.
    :param n: (Optional) The number of completions to generate.
    :param presence_penalty: (Optional) The penalty for repeated tokens.
    :param seed: (Optional) The seed to use.
    :param stop: (Optional) The stop tokens to use.
    :param stream: (Optional) Whether to stream the response.
    :param stream_options: (Optional) The stream options to use.
    :param temperature: (Optional) The temperature to use.
    :param top_p: (Optional) The top p to use.
    :param user: (Optional) The user to use.
    :param suffix: (Optional) The suffix that should be appended to the completion.
    """

    # Standard OpenAI completion parameters
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    best_of: int | None = None
    echo: bool | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
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
    """Request parameters for OpenAI-compatible chat completion endpoint.

    :param model: The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint.
    :param messages: List of messages in the conversation.
    :param frequency_penalty: (Optional) The penalty for repeated tokens.
    :param function_call: (Optional) The function call to use.
    :param functions: (Optional) List of functions to use.
    :param logit_bias: (Optional) The logit bias to use.
    :param logprobs: (Optional) The log probabilities to use.
    :param max_completion_tokens: (Optional) The maximum number of tokens to generate.
    :param max_tokens: (Optional) The maximum number of tokens to generate.
    :param n: (Optional) The number of completions to generate.
    :param parallel_tool_calls: (Optional) Whether to parallelize tool calls.
    :param presence_penalty: (Optional) The penalty for repeated tokens.
    :param response_format: (Optional) The response format to use.
    :param seed: (Optional) The seed to use.
    :param stop: (Optional) The stop tokens to use.
    :param stream: (Optional) Whether to stream the response.
    :param stream_options: (Optional) The stream options to use.
    :param temperature: (Optional) The temperature to use.
    :param tool_choice: (Optional) The tool choice to use.
    :param tools: (Optional) The tools to use.
    :param top_logprobs: (Optional) The top log probabilities to use.
    :param top_p: (Optional) The top p to use.
    :param user: (Optional) The user to use.
    """

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
    """Request parameters for OpenAI-compatible embeddings endpoint.

    :param model: The identifier of the model to use. The model must be an embedding model registered with Llama Stack and available via the /models endpoint.
    :param input: Input text to embed, encoded as a string or array of strings. To embed multiple inputs in a single request, pass an array of strings.
    :param encoding_format: (Optional) The format to return the embeddings in. Can be either "float" or "base64". Defaults to "float".
    :param dimensions: (Optional) The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.
    :param user: (Optional) A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    """

    model: str
    input: str | list[str]
    encoding_format: str | None = "float"
    dimensions: int | None = None
    user: str | None = None


@runtime_checkable
@telemetry_traceable
class InferenceProvider(Protocol):
    """
    This protocol defines the interface that should be implemented by all inference providers.
    """

    API_NAMESPACE: str = "Inference"

    model_store: ModelStore | None = None

    @webmethod(route="/inference/rerank", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def rerank(
        self,
        model: str,
        query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam],
        max_num_results: int | None = None,
    ) -> RerankResponse:
        """Rerank a list of documents based on their relevance to a query.

        :param model: The identifier of the reranking model to use.
        :param query: The search query to rank items against. Can be a string, text content part, or image content part. The input must not exceed the model's max input token length.
        :param items: List of items to rerank. Each item can be a string, text content part, or image content part. Each input must not exceed the model's max input token length.
        :param max_num_results: (Optional) Maximum number of results to return. Default: returns all.
        :returns: RerankResponse with indices sorted by relevance score (descending).
        """
        raise NotImplementedError("Reranking is not implemented")
        return  # this is so mypy's safe-super rule will consider the method concrete

    @webmethod(route="/completions", method="POST", level=LLAMA_STACK_API_V1)
    async def openai_completion(
        self,
        params: Annotated[OpenAICompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAICompletion:
        """Create completion.

        Generate an OpenAI-compatible completion for the given prompt using the specified model.
        :returns: An OpenAICompletion.
        """
        ...

    @webmethod(route="/chat/completions", method="POST", level=LLAMA_STACK_API_V1)
    async def openai_chat_completion(
        self,
        params: Annotated[OpenAIChatCompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Create chat completions.

        Generate an OpenAI-compatible chat completion for the given messages using the specified model.
        :returns: An OpenAIChatCompletion.
        """
        ...

    @webmethod(route="/embeddings", method="POST", level=LLAMA_STACK_API_V1)
    async def openai_embeddings(
        self,
        params: Annotated[OpenAIEmbeddingsRequestWithExtraBody, Body(...)],
    ) -> OpenAIEmbeddingsResponse:
        """Create embeddings.

        Generate OpenAI-compatible embeddings for the given input using the specified model.
        :returns: An OpenAIEmbeddingsResponse containing the embeddings.
        """
        ...


class Inference(InferenceProvider):
    """Inference

    Llama Stack Inference API for generating completions, chat completions, and embeddings.

    This API provides the raw interface to the underlying models. Three kinds of models are supported:
    - LLM models: these models generate "raw" and "chat" (conversational) completions.
    - Embedding models: these models generate embeddings to be used for semantic search.
    - Rerank models: these models reorder the documents based on their relevance to a query.
    """

    @webmethod(route="/chat/completions", method="GET", level=LLAMA_STACK_API_V1)
    async def list_chat_completions(
        self,
        after: str | None = None,
        limit: int | None = 20,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIChatCompletionResponse:
        """List chat completions.

        :param after: The ID of the last chat completion to return.
        :param limit: The maximum number of chat completions to return.
        :param model: The model to filter by.
        :param order: The order to sort the chat completions by: "asc" or "desc". Defaults to "desc".
        :returns: A ListOpenAIChatCompletionResponse.
        """
        raise NotImplementedError("List chat completions is not implemented")

    @webmethod(route="/chat/completions/{completion_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_chat_completion(self, completion_id: str) -> OpenAICompletionWithInputMessages:
        """Get chat completion.

        Describe a chat completion by its ID.

        :param completion_id: ID of the chat completion.
        :returns: A OpenAICompletionWithInputMessages.
        """
        raise NotImplementedError("Get chat completion is not implemented")
