# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from openai.types.chat import ChatCompletionToolChoiceOptionParam as OpenAIChatCompletionToolChoiceOptionParam
from openai.types.chat import ChatCompletionToolParam as OpenAIChatCompletionToolParam
from pydantic import Field, TypeAdapter

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.common.errors import ModelNotFoundError, ModelTypeError
from llama_stack.apis.inference import (
    BatchChatCompletionResponse,
    BatchCompletionResponse,
    ChatCompletionResponse,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    ListOpenAIChatCompletionResponse,
    LogProbConfig,
    Message,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIChoiceLogprobs,
    OpenAICompletion,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    Order,
    ResponseFormat,
    SamplingParams,
    StopReason,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.apis.telemetry import MetricEvent, MetricInResponse, Telemetry
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.providers.datatypes import HealthResponse, HealthStatus, RoutingTable
from llama_stack.providers.utils.inference.inference_store import InferenceStore
from llama_stack.providers.utils.telemetry.tracing import enqueue_event, get_current_span

logger = get_logger(name=__name__, category="core::routers")


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
        telemetry: Telemetry | None = None,
        store: InferenceStore | None = None,
    ) -> None:
        logger.debug("Initializing InferenceRouter")
        self.routing_table = routing_table
        self.telemetry = telemetry
        self.store = store
        if self.telemetry:
            self.tokenizer = Tokenizer.get_instance()
            self.formatter = ChatFormat(self.tokenizer)

    async def initialize(self) -> None:
        logger.debug("InferenceRouter.initialize")

    async def shutdown(self) -> None:
        logger.debug("InferenceRouter.shutdown")
        if self.store:
            try:
                await self.store.shutdown()
            except Exception as e:
                logger.warning(f"Error during InferenceStore shutdown: {e}")

    async def register_model(
        self,
        model_id: str,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
    ) -> None:
        logger.debug(
            f"InferenceRouter.register_model: {model_id=} {provider_model_id=} {provider_id=} {metadata=} {model_type=}",
        )
        await self.routing_table.register_model(model_id, provider_model_id, provider_id, metadata, model_type)

    def _construct_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: Model,
    ) -> list[MetricEvent]:
        """Constructs a list of MetricEvent objects containing token usage metrics.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens used
            model: Model object containing model_id and provider_id

        Returns:
            List of MetricEvent objects with token usage metrics
        """
        span = get_current_span()
        if span is None:
            logger.warning("No span found for token usage metrics")
            return []

        metrics = [
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ]
        metric_events = []
        for metric_name, value in metrics:
            metric_events.append(
                MetricEvent(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    metric=metric_name,
                    value=value,
                    timestamp=datetime.now(UTC),
                    unit="tokens",
                    attributes={
                        "model_id": model.model_id,
                        "provider_id": model.provider_id,
                    },
                )
            )
        return metric_events

    async def _compute_and_log_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: Model,
    ) -> list[MetricInResponse]:
        metrics = self._construct_metrics(prompt_tokens, completion_tokens, total_tokens, model)
        if self.telemetry:
            for metric in metrics:
                enqueue_event(metric)
        return [MetricInResponse(metric=metric.metric, value=metric.value) for metric in metrics]

    async def _count_tokens(
        self,
        messages: list[Message] | InterleavedContent,
        tool_prompt_format: ToolPromptFormat | None = None,
    ) -> int | None:
        if not hasattr(self, "formatter") or self.formatter is None:
            return None

        if isinstance(messages, list):
            encoded = self.formatter.encode_dialog_prompt(messages, tool_prompt_format)
        else:
            encoded = self.formatter.encode_content(messages)
        return len(encoded.tokens) if encoded and encoded.tokens else 0

    async def _get_model(self, model_id: str, expected_model_type: str) -> Model:
        """takes a model id and gets model after ensuring that it is accessible and of the correct type"""
        model = await self.routing_table.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)
        if model.model_type != expected_model_type:
            raise ModelTypeError(model_id, model.model_type, expected_model_type)
        return model

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        tool_prompt_format: ToolPromptFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionResponseStreamChunk]:
        logger.debug(
            f"InferenceRouter.chat_completion: {model_id=}, {stream=}, {messages=}, {tools=}, {tool_config=}, {response_format=}",
        )
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self._get_model(model_id, ModelType.llm)
        if tool_config:
            if tool_choice and tool_choice != tool_config.tool_choice:
                raise ValueError("tool_choice and tool_config.tool_choice must match")
            if tool_prompt_format and tool_prompt_format != tool_config.tool_prompt_format:
                raise ValueError("tool_prompt_format and tool_config.tool_prompt_format must match")
        else:
            params = {}
            if tool_choice:
                params["tool_choice"] = tool_choice
            if tool_prompt_format:
                params["tool_prompt_format"] = tool_prompt_format
            tool_config = ToolConfig(**params)

        tools = tools or []
        if tool_config.tool_choice == ToolChoice.none:
            tools = []
        elif tool_config.tool_choice == ToolChoice.auto:
            pass
        elif tool_config.tool_choice == ToolChoice.required:
            pass
        else:
            # verify tool_choice is one of the tools
            tool_names = [t.tool_name if isinstance(t.tool_name, str) else t.tool_name.value for t in tools]
            if tool_config.tool_choice not in tool_names:
                raise ValueError(f"Tool choice {tool_config.tool_choice} is not one of the tools: {tool_names}")

        params = dict(
            model_id=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )
        provider = await self.routing_table.get_provider_impl(model_id)
        prompt_tokens = await self._count_tokens(messages, tool_config.tool_prompt_format)

        if stream:
            response_stream = await provider.chat_completion(**params)
            return self.stream_tokens_and_compute_metrics(
                response=response_stream,
                prompt_tokens=prompt_tokens,
                model=model,
                tool_prompt_format=tool_config.tool_prompt_format,
            )

        response = await provider.chat_completion(**params)
        metrics = await self.count_tokens_and_compute_metrics(
            response=response,
            prompt_tokens=prompt_tokens,
            model=model,
            tool_prompt_format=tool_config.tool_prompt_format,
        )
        # these metrics will show up in the client response.
        response.metrics = (
            metrics if not hasattr(response, "metrics") or response.metrics is None else response.metrics + metrics
        )
        return response

    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: list[list[Message]],
        tools: list[ToolDefinition] | None = None,
        tool_config: ToolConfig | None = None,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchChatCompletionResponse:
        logger.debug(
            f"InferenceRouter.batch_chat_completion: {model_id=}, {len(messages_batch)=}, {sampling_params=}, {response_format=}, {logprobs=}",
        )
        provider = await self.routing_table.get_provider_impl(model_id)
        return await provider.batch_chat_completion(
            model_id=model_id,
            messages_batch=messages_batch,
            tools=tools,
            tool_config=tool_config,
            sampling_params=sampling_params,
            response_format=response_format,
            logprobs=logprobs,
        )

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        logger.debug(
            f"InferenceRouter.completion: {model_id=}, {stream=}, {content=}, {sampling_params=}, {response_format=}",
        )
        model = await self._get_model(model_id, ModelType.llm)
        provider = await self.routing_table.get_provider_impl(model_id)
        params = dict(
            model_id=model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )

        prompt_tokens = await self._count_tokens(content)
        response = await provider.completion(**params)
        if stream:
            return self.stream_tokens_and_compute_metrics(
                response=response,
                prompt_tokens=prompt_tokens,
                model=model,
            )

        metrics = await self.count_tokens_and_compute_metrics(
            response=response, prompt_tokens=prompt_tokens, model=model
        )
        response.metrics = metrics if response.metrics is None else response.metrics + metrics

        return response

    async def batch_completion(
        self,
        model_id: str,
        content_batch: list[InterleavedContent],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchCompletionResponse:
        logger.debug(
            f"InferenceRouter.batch_completion: {model_id=}, {len(content_batch)=}, {sampling_params=}, {response_format=}, {logprobs=}",
        )
        provider = await self.routing_table.get_provider_impl(model_id)
        return await provider.batch_completion(model_id, content_batch, sampling_params, response_format, logprobs)

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        logger.debug(f"InferenceRouter.embeddings: {model_id}")
        await self._get_model(model_id, ModelType.embedding)
        provider = await self.routing_table.get_provider_impl(model_id)
        return await provider.embeddings(
            model_id=model_id,
            contents=contents,
            text_truncation=text_truncation,
            output_dimension=output_dimension,
            task_type=task_type,
        )

    async def openai_completion(
        self,
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        suffix: str | None = None,
    ) -> OpenAICompletion:
        logger.debug(
            f"InferenceRouter.openai_completion: {model=}, {stream=}, {prompt=}",
        )
        model_obj = await self._get_model(model, ModelType.llm)
        params = dict(
            model=model_obj.identifier,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            user=user,
            guided_choice=guided_choice,
            prompt_logprobs=prompt_logprobs,
            suffix=suffix,
        )
        provider = await self.routing_table.get_provider_impl(model_obj.identifier)
        if stream:
            return await provider.openai_completion(**params)
            # TODO: Metrics do NOT work with openai_completion stream=True due to the fact
            # that we do not return an AsyncIterator, our tests expect a stream of chunks we cannot intercept currently.
            # response_stream = await provider.openai_completion(**params)

        response = await provider.openai_completion(**params)
        if self.telemetry:
            metrics = self._construct_metrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=model_obj,
            )
            for metric in metrics:
                enqueue_event(metric)

            # these metrics will show up in the client response.
            response.metrics = (
                metrics if not hasattr(response, "metrics") or response.metrics is None else response.metrics + metrics
            )
        return response

    async def openai_chat_completion(
        self,
        model: str,
        messages: Annotated[list[OpenAIMessageParam], Field(..., min_length=1)],
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
        logger.debug(
            f"InferenceRouter.openai_chat_completion: {model=}, {stream=}, {messages=}",
        )
        model_obj = await self._get_model(model, ModelType.llm)

        # Use the OpenAI client for a bit of extra input validation without
        # exposing the OpenAI client itself as part of our API surface
        if tool_choice:
            TypeAdapter(OpenAIChatCompletionToolChoiceOptionParam).validate_python(tool_choice)
            if tools is None:
                raise ValueError("'tool_choice' is only allowed when 'tools' is also provided")
        if tools:
            for tool in tools:
                TypeAdapter(OpenAIChatCompletionToolParam).validate_python(tool)

        # Some providers make tool calls even when tool_choice is "none"
        # so just clear them both out to avoid unexpected tool calls
        if tool_choice == "none" and tools is not None:
            tool_choice = None
            tools = None

        params = dict(
            model=model_obj.identifier,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
        )
        provider = await self.routing_table.get_provider_impl(model_obj.identifier)
        if stream:
            response_stream = await provider.openai_chat_completion(**params)

            # For streaming, the provider returns AsyncIterator[OpenAIChatCompletionChunk]
            # We need to add metrics to each chunk and store the final completion
            return self.stream_tokens_and_compute_metrics_openai_chat(
                response=response_stream,
                model=model_obj,
                messages=messages,
            )

        response = await self._nonstream_openai_chat_completion(provider, params)

        # Store the response with the ID that will be returned to the client
        if self.store:
            asyncio.create_task(self.store.store_chat_completion(response, messages))

        if self.telemetry:
            metrics = self._construct_metrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=model_obj,
            )
            for metric in metrics:
                enqueue_event(metric)
            # these metrics will show up in the client response.
            response.metrics = (
                metrics if not hasattr(response, "metrics") or response.metrics is None else response.metrics + metrics
            )
        return response

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        logger.debug(
            f"InferenceRouter.openai_embeddings: {model=}, input_type={type(input)}, {encoding_format=}, {dimensions=}",
        )
        model_obj = await self._get_model(model, ModelType.embedding)
        params = dict(
            model=model_obj.identifier,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )

        provider = await self.routing_table.get_provider_impl(model_obj.identifier)
        return await provider.openai_embeddings(**params)

    async def list_chat_completions(
        self,
        after: str | None = None,
        limit: int | None = 20,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIChatCompletionResponse:
        if self.store:
            return await self.store.list_chat_completions(after, limit, model, order)
        raise NotImplementedError("List chat completions is not supported: inference store is not configured.")

    async def get_chat_completion(self, completion_id: str) -> OpenAICompletionWithInputMessages:
        if self.store:
            return await self.store.get_chat_completion(completion_id)
        raise NotImplementedError("Get chat completion is not supported: inference store is not configured.")

    async def _nonstream_openai_chat_completion(self, provider: Inference, params: dict) -> OpenAIChatCompletion:
        response = await provider.openai_chat_completion(**params)
        for choice in response.choices:
            # some providers return an empty list for no tool calls in non-streaming responses
            # but the OpenAI API returns None. So, set tool_calls to None if it's empty
            if choice.message and choice.message.tool_calls is not None and len(choice.message.tool_calls) == 0:
                choice.message.tool_calls = None
        return response

    async def health(self) -> dict[str, HealthResponse]:
        health_statuses = {}
        timeout = 1  # increasing the timeout to 1 second for health checks
        for provider_id, impl in self.routing_table.impls_by_provider_id.items():
            try:
                # check if the provider has a health method
                if not hasattr(impl, "health"):
                    continue
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                health_statuses[provider_id] = health
            except TimeoutError:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR,
                    message=f"Health check timed out after {timeout} seconds",
                )
            except NotImplementedError:
                health_statuses[provider_id] = HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
            except Exception as e:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"
                )
        return health_statuses

    async def stream_tokens_and_compute_metrics(
        self,
        response,
        prompt_tokens,
        model,
        tool_prompt_format: ToolPromptFormat | None = None,
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None] | AsyncGenerator[CompletionResponseStreamChunk, None]:
        completion_text = ""
        async for chunk in response:
            complete = False
            if hasattr(chunk, "event"):  # only ChatCompletions have .event
                if chunk.event.event_type == ChatCompletionResponseEventType.progress:
                    if chunk.event.delta.type == "text":
                        completion_text += chunk.event.delta.text
                if chunk.event.event_type == ChatCompletionResponseEventType.complete:
                    complete = True
                    completion_tokens = await self._count_tokens(
                        [
                            CompletionMessage(
                                content=completion_text,
                                stop_reason=StopReason.end_of_turn,
                            )
                        ],
                        tool_prompt_format=tool_prompt_format,
                    )
            else:
                if hasattr(chunk, "delta"):
                    completion_text += chunk.delta
                if hasattr(chunk, "stop_reason") and chunk.stop_reason and self.telemetry:
                    complete = True
                    completion_tokens = await self._count_tokens(completion_text)
            # if we are done receiving tokens
            if complete:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

                # Create a separate span for streaming completion metrics
                if self.telemetry:
                    # Log metrics in the new span context
                    completion_metrics = self._construct_metrics(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        model=model,
                    )
                    for metric in completion_metrics:
                        if metric.metric in [
                            "completion_tokens",
                            "total_tokens",
                        ]:  # Only log completion and total tokens
                            enqueue_event(metric)

                        # Return metrics in response
                        async_metrics = [
                            MetricInResponse(metric=metric.metric, value=metric.value) for metric in completion_metrics
                        ]
                        chunk.metrics = async_metrics if chunk.metrics is None else chunk.metrics + async_metrics
                else:
                    # Fallback if no telemetry
                    completion_metrics = self._construct_metrics(
                        prompt_tokens or 0,
                        completion_tokens or 0,
                        total_tokens,
                        model,
                    )
                    async_metrics = [
                        MetricInResponse(metric=metric.metric, value=metric.value) for metric in completion_metrics
                    ]
                    chunk.metrics = async_metrics if chunk.metrics is None else chunk.metrics + async_metrics
            yield chunk

    async def count_tokens_and_compute_metrics(
        self,
        response: ChatCompletionResponse | CompletionResponse,
        prompt_tokens,
        model,
        tool_prompt_format: ToolPromptFormat | None = None,
    ):
        if isinstance(response, ChatCompletionResponse):
            content = [response.completion_message]
        else:
            content = response.content
        completion_tokens = await self._count_tokens(messages=content, tool_prompt_format=tool_prompt_format)
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        # Create a separate span for completion metrics
        if self.telemetry:
            # Log metrics in the new span context
            completion_metrics = self._construct_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model,
            )
            for metric in completion_metrics:
                if metric.metric in ["completion_tokens", "total_tokens"]:  # Only log completion and total tokens
                    enqueue_event(metric)

            # Return metrics in response
            return [MetricInResponse(metric=metric.metric, value=metric.value) for metric in completion_metrics]

        # Fallback if no telemetry
        metrics = self._construct_metrics(
            prompt_tokens or 0,
            completion_tokens or 0,
            total_tokens,
            model,
        )
        return [MetricInResponse(metric=metric.metric, value=metric.value) for metric in metrics]

    async def stream_tokens_and_compute_metrics_openai_chat(
        self,
        response: AsyncIterator[OpenAIChatCompletionChunk],
        model: Model,
        messages: list[OpenAIMessageParam] | None = None,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        """Stream OpenAI chat completion chunks, compute metrics, and store the final completion."""
        id = None
        created = None
        choices_data: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in response:
                # Skip None chunks
                if chunk is None:
                    continue

                # Capture ID and created timestamp from first chunk
                if id is None and chunk.id:
                    id = chunk.id
                if created is None and chunk.created:
                    created = chunk.created

                # Accumulate choice data for final assembly
                if chunk.choices:
                    for choice_delta in chunk.choices:
                        idx = choice_delta.index
                        if idx not in choices_data:
                            choices_data[idx] = {
                                "content_parts": [],
                                "tool_calls_builder": {},
                                "finish_reason": "stop",
                                "logprobs_content_parts": [],
                            }
                        current_choice_data = choices_data[idx]

                        if choice_delta.delta:
                            delta = choice_delta.delta
                            if delta.content:
                                current_choice_data["content_parts"].append(delta.content)
                            if delta.tool_calls:
                                for tool_call_delta in delta.tool_calls:
                                    tc_idx = tool_call_delta.index
                                    if tc_idx not in current_choice_data["tool_calls_builder"]:
                                        current_choice_data["tool_calls_builder"][tc_idx] = {
                                            "id": None,
                                            "type": "function",
                                            "function_name_parts": [],
                                            "function_arguments_parts": [],
                                        }
                                    builder = current_choice_data["tool_calls_builder"][tc_idx]
                                    if tool_call_delta.id:
                                        builder["id"] = tool_call_delta.id
                                    if tool_call_delta.type:
                                        builder["type"] = tool_call_delta.type
                                    if tool_call_delta.function:
                                        if tool_call_delta.function.name:
                                            builder["function_name_parts"].append(tool_call_delta.function.name)
                                        if tool_call_delta.function.arguments:
                                            builder["function_arguments_parts"].append(
                                                tool_call_delta.function.arguments
                                            )
                        if choice_delta.finish_reason:
                            current_choice_data["finish_reason"] = choice_delta.finish_reason
                        if choice_delta.logprobs and choice_delta.logprobs.content:
                            current_choice_data["logprobs_content_parts"].extend(choice_delta.logprobs.content)

                # Compute metrics on final chunk
                if chunk.choices and chunk.choices[0].finish_reason:
                    completion_text = ""
                    for choice_data in choices_data.values():
                        completion_text += "".join(choice_data["content_parts"])

                    # Add metrics to the chunk
                    if self.telemetry and chunk.usage:
                        metrics = self._construct_metrics(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                            model=model,
                        )
                        for metric in metrics:
                            enqueue_event(metric)

                yield chunk
        finally:
            # Store the final assembled completion
            if id and self.store and messages:
                assembled_choices: list[OpenAIChoice] = []
                for choice_idx, choice_data in choices_data.items():
                    content_str = "".join(choice_data["content_parts"])
                    assembled_tool_calls: list[OpenAIChatCompletionToolCall] = []
                    if choice_data["tool_calls_builder"]:
                        for tc_build_data in choice_data["tool_calls_builder"].values():
                            if tc_build_data["id"]:
                                func_name = "".join(tc_build_data["function_name_parts"])
                                func_args = "".join(tc_build_data["function_arguments_parts"])
                                assembled_tool_calls.append(
                                    OpenAIChatCompletionToolCall(
                                        id=tc_build_data["id"],
                                        type=tc_build_data["type"],
                                        function=OpenAIChatCompletionToolCallFunction(
                                            name=func_name, arguments=func_args
                                        ),
                                    )
                                )
                    message = OpenAIAssistantMessageParam(
                        role="assistant",
                        content=content_str if content_str else None,
                        tool_calls=assembled_tool_calls if assembled_tool_calls else None,
                    )
                    logprobs_content = choice_data["logprobs_content_parts"]
                    final_logprobs = OpenAIChoiceLogprobs(content=logprobs_content) if logprobs_content else None

                    assembled_choices.append(
                        OpenAIChoice(
                            finish_reason=choice_data["finish_reason"],
                            index=choice_idx,
                            message=message,
                            logprobs=final_logprobs,
                        )
                    )

                final_response = OpenAIChatCompletion(
                    id=id,
                    choices=assembled_choices,
                    created=created or int(time.time()),
                    model=model.identifier,
                    object="chat.completion",
                )
                logger.debug(f"InferenceRouter.completion_response: {final_response}")
                asyncio.create_task(self.store.store_chat_completion(final_response, messages))
