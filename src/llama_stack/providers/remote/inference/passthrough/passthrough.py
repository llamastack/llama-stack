# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import AsyncIterator
from enum import Enum

import httpx
from pydantic import BaseModel

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
from llama_stack.core.request_headers import NeedsRequestProviderData

from .config import PassthroughImplConfig


class PassthroughInferenceAdapter(NeedsRequestProviderData, Inference):
    def __init__(self, config: PassthroughImplConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        return model

    async def list_models(self) -> list[Model]:
        """List models by calling the downstream /v1/models endpoint."""
        base_url = self._get_passthrough_url().rstrip("/")
        api_key = self._get_passthrough_api_key()

        url = f"{base_url}/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()

            response_data = response.json()
            models_data = response_data["data"]

            # Convert from OpenAI format to Llama Stack Model format
            models = []
            for model_data in models_data:
                downstream_model_id = model_data["id"]
                custom_metadata = model_data.get("custom_metadata", {})

                # Prefix identifier with provider ID for local registry
                local_identifier = f"{self.__provider_id__}/{downstream_model_id}"

                model = Model(
                    identifier=local_identifier,
                    provider_id=self.__provider_id__,
                    provider_resource_id=downstream_model_id,
                    model_type=custom_metadata.get("model_type", "llm"),
                    metadata=custom_metadata,
                )
                models.append(model)

            return models

    async def should_refresh_models(self) -> bool:
        """Passthrough should refresh models since they come from downstream dynamically."""
        return self.config.refresh_models

    def _get_passthrough_url(self) -> str:
        """Get the passthrough URL from config or provider data."""
        if self.config.url is not None:
            return self.config.url

        provider_data = self.get_request_provider_data()
        if provider_data is None:
            raise ValueError(
                'Pass url of the passthrough endpoint in the header X-LlamaStack-Provider-Data as { "passthrough_url": <your passthrough url>}'
            )
        return provider_data.passthrough_url

    def _get_passthrough_api_key(self) -> str:
        """Get the passthrough API key from config or provider data."""
        if self.config.auth_credential is not None:
            return self.config.auth_credential.get_secret_value()

        provider_data = self.get_request_provider_data()
        if provider_data is None:
            raise ValueError(
                'Pass API Key for the passthrough endpoint in the header X-LlamaStack-Provider-Data as { "passthrough_api_key": <your api key>}'
            )
        return provider_data.passthrough_api_key

    def _serialize_value(self, value):
        """Convert Pydantic models and enums to JSON-serializable values."""
        if isinstance(value, BaseModel):
            return json.loads(value.model_dump_json())
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

    async def _make_request(
        self,
        endpoint: str,
        params: dict,
        response_type: type,
        stream: bool = False,
    ):
        """Make an HTTP request to the passthrough endpoint."""
        base_url = self._get_passthrough_url().rstrip("/")
        api_key = self._get_passthrough_api_key()

        url = f"{base_url}/v1/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Serialize the request body
        json_body = self._serialize_value(params)

        if stream:
            return self._stream_request(url, headers, json_body, response_type)
        else:
            return await self._non_stream_request(url, headers, json_body, response_type)

    async def _non_stream_request(
        self,
        url: str,
        headers: dict,
        json_body: dict,
        response_type: type,
    ):
        """Make a non-streaming HTTP request."""
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_body, headers=headers, timeout=30.0)
            response.raise_for_status()

            response_data = response.json()
            return response_type.model_validate(response_data)

    async def _stream_request(
        self,
        url: str,
        headers: dict,
        json_body: dict,
        response_type: type,
    ) -> AsyncIterator:
        """Make a streaming HTTP request with Server-Sent Events parsing."""
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=json_body, headers=headers, timeout=30.0) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        # Extract JSON after "data: " prefix
                        data_str = line[len("data:") :].strip()

                        # Skip empty lines or "[DONE]" marker
                        if not data_str or data_str == "[DONE]":
                            continue

                        data = json.loads(data_str)

                        # Fix OpenAI compatibility: finish_reason can be null in intermediate chunks
                        # but our Pydantic model may not accept null
                        if "choices" in data:
                            for choice in data["choices"]:
                                if choice.get("finish_reason") is None:
                                    choice["finish_reason"] = ""

                        yield response_type.model_validate(data)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        # params.model is already the provider_resource_id (router translated it)
        request_params = params.model_dump(exclude_none=True)

        return await self._make_request(
            endpoint="completions",
            params=request_params,
            response_type=OpenAICompletion,
            stream=params.stream or False,
        )

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        # params.model is already the provider_resource_id (router translated it)
        request_params = params.model_dump(exclude_none=True)

        return await self._make_request(
            endpoint="chat/completions",
            params=request_params,
            response_type=OpenAIChatCompletionChunk if params.stream else OpenAIChatCompletion,
            stream=params.stream or False,
        )

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        # params.model is already the provider_resource_id (router translated it)
        request_params = params.model_dump(exclude_none=True)

        return await self._make_request(
            endpoint="embeddings",
            params=request_params,
            response_type=OpenAIEmbeddingsResponse,
            stream=False,
        )
