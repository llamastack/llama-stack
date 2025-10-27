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
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.apis.models import Model
from llama_stack.core.library_client import convert_pydantic_to_json_value
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

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
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        client = self._get_client()
        model_obj = await self.model_store.get_model(params.model)

        params = params.model_copy()
        params.model = model_obj.provider_resource_id

        request_params = params.model_dump(exclude_none=True)

        return await client.inference.openai_completion(**request_params)

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        client = self._get_client()
        model_obj = await self.model_store.get_model(params.model)

        params = params.model_copy()
        params.model = model_obj.provider_resource_id

        request_params = params.model_dump(exclude_none=True)

        return await client.inference.openai_chat_completion(**request_params)

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
