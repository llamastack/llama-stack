# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
import asyncio
from typing import Any

from openai import AsyncOpenAI

from llama_stack.apis.inference import *
from llama_stack.apis.inference import (
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.common.content_types import InterleavedContentItem
from llama_stack.apis.models import Model, ModelType
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
    interleaved_content_as_str,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from .config import RunpodImplConfig

MODEL_ENTRIES = []


class RunpodInferenceAdapter(
    OpenAIMixin,
    ModelRegistryHelper,
    Inference,
):
    """
    Adapter for RunPod's OpenAI-compatible API endpoints.
    Supports VLLM for serverless endpoint self-hosted or public endpoints.
    Can work with any runpod endpoints that support OpenAI-compatible API
    """

    def __init__(self, config: RunpodImplConfig) -> None:
        OpenAIMixin.__init__(self)
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)
        self.config = config

    def get_api_key(self) -> str:
        """Get API key for OpenAI client."""
        return self.config.api_token

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return self.config.url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def get_extra_client_params(self) -> dict[str, Any]:
        """Override to add RunPod-specific client parameters if needed."""
        return {}

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
    ):
        """Override to add RunPod-specific stream_options requirement."""
        if stream and not stream_options:
            stream_options = {"include_usage": True}

        return await super().openai_chat_completion(
            model=model,
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

    async def register_model(self, model: Model) -> Model:
        """
        Pass-through registration - accepts any model that the RunPod endpoint serves.
        In the .yaml file the model: can be defined as example
        models:
            - metadata: {}
            model_id: qwen3-32b-awq
            model_type: llm
            provider_id: runpod
            provider_model_id: Qwen/Qwen3-32B-AWQ
        """
        return model

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | AsyncGenerator[CompletionResponseStreamChunk, None]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Resolve model_id to provider_resource_id
        model = await self.model_store.get_model(model_id)
        provider_model_id = model.provider_resource_id or model_id

        request = CompletionRequest(
            model=provider_model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )

        if stream:
            return self._stream_completion(request, self.client)
        else:
            return await self._nonstream_completion(request, self.client)

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        """Process chat completion requests using RunPod's OpenAI-compatible API."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Resolve model_id to provider_resource_id
        model = await self.model_store.get_model(model_id)
        provider_model_id = model.provider_resource_id or model_id

        request = ChatCompletionRequest(
            model=provider_model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request, self.client)
        else:
            return await self._nonstream_chat_completion(request, self.client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> ChatCompletionResponse:
        params = await self._get_chat_params(request)
        # Make actual RunPod API call
        r = await client.chat.completions.create(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        params = await self._get_chat_params(request)
        # Make actual RunPod API call for streaming
        stream = await client.chat.completions.create(**params)
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_chat_params(self, request: ChatCompletionRequest) -> dict:
        """Convert Llama Stack request to RunPod API parameters."""
        messages = [await convert_message_to_openai_dict(m, download=False) for m in request.messages]

        params = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

        if request.stream:
            params["stream_options"] = {"include_usage": True}

        return params

    async def _nonstream_completion(
        self, request: CompletionRequest, client: AsyncOpenAI
    ) -> CompletionResponse:
        params = await self._get_completion_params(request)
        # Make actual RunPod API call
        r = await client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(
        self, request: CompletionRequest, client: AsyncOpenAI
    ) -> AsyncGenerator:
        params = await self._get_completion_params(request)
        # Make actual RunPod API call for streaming
        stream = await client.completions.create(**params)
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def _get_completion_params(self, request: CompletionRequest) -> dict:
        """Convert Llama Stack request to RunPod API parameters."""
        params = {
            "model": request.model,
            "prompt": await completion_request_to_prompt(request),
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

        if request.stream:
            params["stream_options"] = {"include_usage": True}

        return params

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(model_id)
        model = model_obj.provider_resource_id or model_id

        kwargs = {}
        if output_dimension:
            kwargs["dimensions"] = output_dimension

        response = await self.client.embeddings.create(
            model=model,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(model)
        provider_model_id = model_obj.provider_resource_id or model

        response = await self.client.embeddings.create(
            model=provider_model_id,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )

        return response
