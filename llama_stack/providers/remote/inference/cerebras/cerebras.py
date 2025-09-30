# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from urllib.parse import urljoin

from cerebras.cloud.sdk import AsyncCerebras

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    Inference,
    LogProbConfig,
    Message,
    OpenAIEmbeddingsResponse,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
    TopKSamplingStrategy,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
)

from .config import CerebrasImplConfig


class CerebrasInferenceAdapter(
    OpenAIMixin,
    ModelRegistryHelper,
    Inference,
):
    def __init__(self, config: CerebrasImplConfig) -> None:
        self.config = config

        # TODO: make this use provider data, etc. like other providers
        self._cerebras_client = AsyncCerebras(
            base_url=self.config.base_url,
            api_key=self.config.api_key.get_secret_value(),
        )

    def get_api_key(self) -> str:
        return self.config.api_key.get_secret_value()

    def get_base_url(self) -> str:
        return urljoin(self.config.base_url, "v1")

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        params = await self._get_params(request)

        r = await self._cerebras_client.completions.create(**params)

        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        stream = await self._cerebras_client.completions.create(**params)

        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: ChatCompletionRequest | CompletionRequest) -> dict:
        if request.sampling_params and isinstance(request.sampling_params.strategy, TopKSamplingStrategy):
            raise ValueError("`top_k` not supported by Cerebras")

        prompt = ""
        if isinstance(request, ChatCompletionRequest):
            prompt = await chat_completion_request_to_prompt(request, self.get_llama_model(request.model))
        elif isinstance(request, CompletionRequest):
            prompt = await completion_request_to_prompt(request)
        else:
            raise ValueError(f"Unknown request type {type(request)}")

        return {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
