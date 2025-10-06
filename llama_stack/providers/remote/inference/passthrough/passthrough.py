# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

from llama_stack_client import AsyncLlamaStackClient

from llama_stack.apis.inference import (
    Inference,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.models import Model
from llama_stack.core.library_client import convert_pydantic_to_json_value
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params

from .config import PassthroughImplConfig


class PassthroughInferenceAdapter(Inference):
    def __init__(self, config: PassthroughImplConfig) -> None:
        ModelRegistryHelper.__init__(self)
        self.config = config

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        return model

    def _get_client(self) -> AsyncLlamaStackClient:
        passthrough_url = None
        passthrough_api_key = None
        provider_data = None

        if self.config.url is not None:
            passthrough_url = self.config.url
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.passthrough_url:
                raise ValueError(
                    'Pass url of the passthrough endpoint in the header X-LlamaStack-Provider-Data as { "passthrough_url": <your passthrough url>}'
                )
            passthrough_url = provider_data.passthrough_url

        if self.config.api_key is not None:
            passthrough_api_key = self.config.api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.passthrough_api_key:
                raise ValueError(
                    'Pass API Key for the passthrough endpoint in the header X-LlamaStack-Provider-Data as { "passthrough_api_key": <your api key>}'
                )
            passthrough_api_key = provider_data.passthrough_api_key

        return AsyncLlamaStackClient(
            base_url=passthrough_url,
            api_key=passthrough_api_key,
            provider_data=provider_data,
        )

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
        client = self._get_client()
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
            guided_choice=guided_choice,
            prompt_logprobs=prompt_logprobs,
        )

        return await client.inference.openai_completion(**params)

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
        client = self._get_client()
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

        return await client.inference.openai_chat_completion(**params)

    def cast_value_to_json_dict(self, request_params: dict[str, Any]) -> dict[str, Any]:
        json_params = {}
        for key, value in request_params.items():
            json_input = convert_pydantic_to_json_value(value)
            if isinstance(json_input, dict):
                json_input = {k: v for k, v in json_input.items() if v is not None}
            elif isinstance(json_input, list):
                json_input = [x for x in json_input if x is not None]
                new_input = []
                for x in json_input:
                    if isinstance(x, dict):
                        x = {k: v for k, v in x.items() if v is not None}
                    new_input.append(x)
                json_input = new_input

            json_params[key] = json_input

        return json_params
