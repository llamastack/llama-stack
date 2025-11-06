# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

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
        client = self._get_openai_client()

        response = await client.models.list()

        # Convert from OpenAI format to Llama Stack Model format
        models = []
        for model_data in response.data:
            downstream_model_id = model_data.id
            custom_metadata = getattr(model_data, "custom_metadata", {}) or {}

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

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get an AsyncOpenAI client configured for the downstream server."""
        base_url = self._get_passthrough_url()
        api_key = self._get_passthrough_api_key()

        return AsyncOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=api_key,
        )

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

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        """Forward completion request to downstream using OpenAI client."""
        client = self._get_openai_client()
        request_params = params.model_dump(exclude_none=True)
        response = await client.completions.create(**request_params)
        return response  # type: ignore

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Forward chat completion request to downstream using OpenAI client."""
        client = self._get_openai_client()
        request_params = params.model_dump(exclude_none=True)
        response = await client.chat.completions.create(**request_params)
        return response  # type: ignore

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Forward embeddings request to downstream using OpenAI client."""
        client = self._get_openai_client()
        request_params = params.model_dump(exclude_none=True)
        response = await client.embeddings.create(**request_params)
        return response  # type: ignore
