# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from openai import AsyncOpenAI

from llama_stack.apis.common.content_types import InterleavedContent, InterleavedContentItem
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    EmbeddingsResponse,
    EmbeddingTaskType,
    GreedySamplingStrategy,
    Inference,
    LogProbConfig,
    Message,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    prepare_openai_completion_params,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    request_has_media,
)

from . import WatsonXConfig
from .models import MODEL_ENTRIES


class WatsonXInferenceAdapter(Inference, ModelRegistryHelper):
    def __init__(self, config: WatsonXConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)

        print(f"Initializing watsonx InferenceAdapter({config.url})...")

        self._config = config

        self._project_id = self._config.project_id

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

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
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    def _get_client(self, model_id) -> Model:
        config_api_key = self._config.api_key.get_secret_value() if self._config.api_key else None
        config_url = self._config.url
        project_id = self._config.project_id
        credentials = {"url": config_url, "apikey": config_api_key}

        return Model(model_id=model_id, credentials=credentials, project_id=project_id)

    def _get_openai_client(self) -> AsyncOpenAI:
        if not self._openai_client:
            self._openai_client = AsyncOpenAI(
                base_url=f"{self._config.url}/openai/v1",
                api_key=self._config.api_key,
            )
        return self._openai_client

    async def _nonstream_completion(self, request: CompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = self._get_client(request.model).generate(**params)
        choices = []
        if "results" in r:
            for result in r["results"]:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=result["stop_reason"] if result["stop_reason"] else None,
                    text=result["generated_text"],
                )
                choices.append(choice)
        response = OpenAICompatCompletionResponse(
            choices=choices,
        )
        return process_completion_response(response)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            s = self._get_client(request.model).generate_text_stream(**params)
            for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=None,
                    text=chunk,
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

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
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = self._get_client(request.model).generate(**params)
        choices = []
        if "results" in r:
            for result in r["results"]:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=result["stop_reason"] if result["stop_reason"] else None,
                    text=result["generated_text"],
                )
                choices.append(choice)
        response = OpenAICompatCompletionResponse(
            choices=choices,
        )
        return process_chat_completion_response(response, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)
        model_id = request.model

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            s = self._get_client(model_id).generate_text_stream(**params)
            for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=None,
                    text=chunk,
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: ChatCompletionRequest | CompletionRequest) -> dict:
        input_dict = {"params": {}}
        media_present = request_has_media(request)
        llama_model = self.get_llama_model(request.model)
        if isinstance(request, ChatCompletionRequest):
            input_dict["prompt"] = await chat_completion_request_to_prompt(request, llama_model)
        else:
            assert not media_present, "Together does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)
        if request.sampling_params:
            if request.sampling_params.strategy:
                input_dict["params"][GenParams.DECODING_METHOD] = request.sampling_params.strategy.type
            if request.sampling_params.max_tokens:
                input_dict["params"][GenParams.MAX_NEW_TOKENS] = request.sampling_params.max_tokens
            if request.sampling_params.repetition_penalty:
                input_dict["params"][GenParams.REPETITION_PENALTY] = request.sampling_params.repetition_penalty

            if isinstance(request.sampling_params.strategy, TopPSamplingStrategy):
                input_dict["params"][GenParams.TOP_P] = request.sampling_params.strategy.top_p
                input_dict["params"][GenParams.TEMPERATURE] = request.sampling_params.strategy.temperature
            if isinstance(request.sampling_params.strategy, TopKSamplingStrategy):
                input_dict["params"][GenParams.TOP_K] = request.sampling_params.strategy.top_k
            if isinstance(request.sampling_params.strategy, GreedySamplingStrategy):
                input_dict["params"][GenParams.TEMPERATURE] = 0.0

        input_dict["params"][GenParams.STOP_SEQUENCES] = ["<|endoftext|>"]

        params = {
            **input_dict,
        }
        return params

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        raise NotImplementedError("embedding is not supported for watsonx")

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()

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
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
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
        )
        return await self._get_openai_client().completions.create(**params)  # type: ignore

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
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
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
        if params.get("stream", False):
            return self._stream_openai_chat_completion(params)
        return await self._get_openai_client().chat.completions.create(**params)  # type: ignore

    async def _stream_openai_chat_completion(self, params: dict) -> AsyncGenerator:
        # watsonx.ai sometimes adds usage data to the stream
        include_usage = False
        if params.get("stream_options", None):
            include_usage = params["stream_options"].get("include_usage", False)
        stream = await self._get_openai_client().chat.completions.create(**params)

        seen_finish_reason = False
        async for chunk in stream:
            # Final usage chunk with no choices that the user didn't request, so discard
            if not include_usage and seen_finish_reason and len(chunk.choices) == 0:
                break
            yield chunk
            for choice in chunk.choices:
                if choice.finish_reason:
                    seen_finish_reason = True
                    break
