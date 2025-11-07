# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Body
from openai.types.chat import ChatCompletionToolChoiceOptionParam as OpenAIChatCompletionToolChoiceOptionParam
from openai.types.chat import ChatCompletionToolParam as OpenAIChatCompletionToolParam
from pydantic import TypeAdapter

from llama_stack.apis.common.errors import ModelNotFoundError, ModelTypeError
from llama_stack.apis.inference import (
    Inference,
    ListOpenAIChatCompletionResponse,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIChoiceLogprobs,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    Order,
    RerankResponse,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
)
from llama_stack.apis.models import ModelType
from llama_stack.core.telemetry.telemetry import MetricEvent
from llama_stack.core.telemetry.tracing import enqueue_event, get_current_span
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.providers.datatypes import HealthResponse, HealthStatus, RoutingTable
from llama_stack.providers.utils.inference.inference_store import InferenceStore

logger = get_logger(name=__name__, category="core::routers")


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
        store: InferenceStore | None = None,
        telemetry_enabled: bool = False,
    ) -> None:
        logger.debug("Initializing InferenceRouter")
        self.routing_table = routing_table
        self.telemetry_enabled = telemetry_enabled
        self.store = store
        if self.telemetry_enabled:
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
        fully_qualified_model_id: str,
        provider_id: str,
    ) -> list[MetricEvent]:
        """Constructs a list of MetricEvent objects containing token usage metrics.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens used
            fully_qualified_model_id:
            provider_id: The provider identifier

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
                        "model_id": fully_qualified_model_id,
                        "provider_id": provider_id,
                    },
                )
            )
        return metric_events

    async def _get_model_provider(self, model_id: str, expected_model_type: str) -> tuple[Inference, str]:
        model = await self.routing_table.get_object_by_identifier("model", model_id)
        if model:
            if model.model_type != expected_model_type:
                raise ModelTypeError(model_id, model.model_type, expected_model_type)

            provider = await self.routing_table.get_provider_impl(model.identifier)
            return provider, model.provider_resource_id

        splits = model_id.split("/", maxsplit=1)
        if len(splits) != 2:
            raise ModelNotFoundError(model_id)

        provider_id, provider_resource_id = splits
        if provider_id not in self.routing_table.impls_by_provider_id:
            logger.warning(f"Provider {provider_id} not found for model {model_id}")
            raise ModelNotFoundError(model_id)

        return self.routing_table.impls_by_provider_id[provider_id], provider_resource_id

    async def rerank(
        self,
        model: str,
        query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam],
        max_num_results: int | None = None,
    ) -> RerankResponse:
        logger.debug(f"InferenceRouter.rerank: {model}")
        provider, provider_resource_id = await self._get_model_provider(model, ModelType.rerank)
        return await provider.rerank(provider_resource_id, query, items, max_num_results)

    async def openai_completion(
        self,
        params: Annotated[OpenAICompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAICompletion:
        logger.debug(
            f"InferenceRouter.openai_completion: model={params.model}, stream={params.stream}, prompt={params.prompt}",
        )
        request_model_id = params.model
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.llm)
        params.model = provider_resource_id

        if params.stream:
            return await provider.openai_completion(params)
            # TODO: Metrics do NOT work with openai_completion stream=True due to the fact
            # that we do not return an AsyncIterator, our tests expect a stream of chunks we cannot intercept currently.

        response = await provider.openai_completion(params)
        response.model = request_model_id
        if self.telemetry_enabled and response.usage is not None:
            metrics = self._construct_metrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                fully_qualified_model_id=request_model_id,
                provider_id=provider.__provider_id__,
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
        params: Annotated[OpenAIChatCompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        logger.debug(
            f"InferenceRouter.openai_chat_completion: model={params.model}, stream={params.stream}, messages={params.messages}",
        )
        request_model_id = params.model
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.llm)
        params.model = provider_resource_id

        # Use the OpenAI client for a bit of extra input validation without
        # exposing the OpenAI client itself as part of our API surface
        if params.tool_choice:
            TypeAdapter(OpenAIChatCompletionToolChoiceOptionParam).validate_python(params.tool_choice)
            if params.tools is None:
                raise ValueError("'tool_choice' is only allowed when 'tools' is also provided")
        if params.tools:
            for tool in params.tools:
                TypeAdapter(OpenAIChatCompletionToolParam).validate_python(tool)

        # Some providers make tool calls even when tool_choice is "none"
        # so just clear them both out to avoid unexpected tool calls
        if params.tool_choice == "none" and params.tools is not None:
            params.tool_choice = None
            params.tools = None

        if params.stream:
            response_stream = await provider.openai_chat_completion(params)

            # For streaming, the provider returns AsyncIterator[OpenAIChatCompletionChunk]
            # We need to add metrics to each chunk and store the final completion
            return self.stream_tokens_and_compute_metrics_openai_chat(
                response=response_stream,
                fully_qualified_model_id=request_model_id,
                provider_id=provider.__provider_id__,
                messages=params.messages,
            )

        response = await self._nonstream_openai_chat_completion(provider, params)
        response.model = request_model_id

        # Store the response with the ID that will be returned to the client
        if self.store:
            asyncio.create_task(self.store.store_chat_completion(response, params.messages))

        if self.telemetry_enabled and response.usage is not None:
            metrics = self._construct_metrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                fully_qualified_model_id=request_model_id,
                provider_id=provider.__provider_id__,
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
        params: Annotated[OpenAIEmbeddingsRequestWithExtraBody, Body(...)],
    ) -> OpenAIEmbeddingsResponse:
        logger.debug(
            f"InferenceRouter.openai_embeddings: model={params.model}, input_type={type(params.input)}, encoding_format={params.encoding_format}, dimensions={params.dimensions}",
        )
        request_model_id = params.model
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.embedding)
        params.model = provider_resource_id

        response = await provider.openai_embeddings(params)
        response.model = request_model_id
        return response

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

    async def _nonstream_openai_chat_completion(
        self, provider: Inference, params: OpenAIChatCompletionRequestWithExtraBody
    ) -> OpenAIChatCompletion:
        response = await provider.openai_chat_completion(params)
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

    async def stream_tokens_and_compute_metrics_openai_chat(
        self,
        response: AsyncIterator[OpenAIChatCompletionChunk],
        fully_qualified_model_id: str,
        provider_id: str,
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

                chunk.model = fully_qualified_model_id

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
                    if self.telemetry_enabled and hasattr(chunk, "usage") and chunk.usage:
                        metrics = self._construct_metrics(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                            model_id=fully_qualified_model_id,
                            provider_id=provider_id,
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
                    model=fully_qualified_model_id,
                    object="chat.completion",
                )
                logger.debug(f"InferenceRouter.completion_response: {final_response}")
                asyncio.create_task(self.store.store_chat_completion(final_response, messages))
