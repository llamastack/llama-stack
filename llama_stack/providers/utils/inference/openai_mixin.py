# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI

from llama_stack.apis.inference import (
    Model,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params
from llama_stack.providers.utils.inference.prompt_adapter import localize_image_content

logger = get_logger(name=__name__, category="providers::utils")


class OpenAIMixin(ModelRegistryHelper, ABC):
    """
    Mixin class that provides OpenAI-specific functionality for inference providers.
    This class handles direct OpenAI API calls using the AsyncOpenAI client.

    This is an abstract base class that requires child classes to implement:
    - get_api_key(): Method to retrieve the API key
    - get_base_url(): Method to retrieve the OpenAI-compatible API base URL

    Expected Dependencies:
    - self.model_store: Injected by the Llama Stack distribution system at runtime.
      This provides model registry functionality for looking up registered models.
      The model_store is set in routing_tables/common.py during provider initialization.
    """

    # Allow subclasses to control whether to overwrite the 'id' field in OpenAI responses
    # is overwritten with a client-side generated id.
    #
    # This is useful for providers that do not return a unique id in the response.
    overwrite_completion_id: bool = False

    # Allow subclasses to control whether to download images and convert to base64
    # for providers that require base64 encoded images instead of URLs.
    download_images: bool = False

    # Embedding model metadata for this provider
    # Can be set by subclasses or instances to provide embedding models
    # Format: {"model_id": {"embedding_dimension": 1536, "context_length": 8192}}
    embedding_model_metadata: dict[str, dict[str, int]] = {}

    # Cache of available models keyed by model ID
    # This is set in list_models() and used in check_model_availability()
    _model_cache: dict[str, Model] = {}

    # List of allowed models for this provider, if empty all models allowed
    allowed_models: list[str] = []

    @abstractmethod
    def get_api_key(self) -> str:
        """
        Get the API key.

        This method must be implemented by child classes to provide the API key
        for authenticating with the OpenAI API or compatible endpoints.

        :return: The API key as a string
        """
        pass

    @abstractmethod
    def get_base_url(self) -> str:
        """
        Get the OpenAI-compatible API base URL.

        This method must be implemented by child classes to provide the base URL
        for the OpenAI API or compatible endpoints (e.g., "https://api.openai.com/v1").

        :return: The base URL as a string
        """
        pass

    def get_extra_client_params(self) -> dict[str, Any]:
        """
        Get any extra parameters to pass to the AsyncOpenAI client.

        Child classes can override this method to provide additional parameters
        such as timeout settings, proxies, etc.

        :return: A dictionary of extra parameters
        """
        return {}

    @property
    def client(self) -> AsyncOpenAI:
        """
        Get an AsyncOpenAI client instance.

        Uses the abstract methods get_api_key() and get_base_url() which must be
        implemented by child classes.
        """
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
            **self.get_extra_client_params(),
        )

    async def _get_provider_model_id(self, model: str) -> str:
        """
        Get the provider-specific model ID from the model store.

        This is a utility method that looks up the registered model and returns
        the provider_resource_id that should be used for actual API calls.

        :param model: The registered model name/identifier
        :return: The provider-specific model ID (e.g., "gpt-4")
        """
        # Look up the registered model to get the provider-specific model ID
        # self.model_store is injected by the distribution system at runtime
        model_obj: Model = await self.model_store.get_model(model)  # type: ignore[attr-defined]
        # provider_resource_id is str | None, but we expect it to be str for OpenAI calls
        if model_obj.provider_resource_id is None:
            raise ValueError(f"Model {model} has no provider_resource_id")
        return model_obj.provider_resource_id

    async def _maybe_overwrite_id(self, resp: Any, stream: bool | None) -> Any:
        if not self.overwrite_completion_id:
            return resp

        new_id = f"cltsd-{uuid.uuid4()}"
        if stream:

            async def _gen():
                async for chunk in resp:
                    chunk.id = new_id
                    yield chunk

            return _gen()
        else:
            resp.id = new_id
            return resp

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
        """
        Direct OpenAI completion API call.
        """
        # Handle parameters that are not supported by OpenAI API, but may be by the provider
        #  prompt_logprobs is supported by vLLM
        #  guided_choice is supported by vLLM
        # TODO: test coverage
        extra_body: dict[str, Any] = {}
        if prompt_logprobs is not None and prompt_logprobs >= 0:
            extra_body["prompt_logprobs"] = prompt_logprobs
        if guided_choice:
            extra_body["guided_choice"] = guided_choice

        # TODO: fix openai_completion to return type compatible with OpenAI's API response
        resp = await self.client.completions.create(
            **await prepare_openai_completion_params(
                model=await self._get_provider_model_id(model),
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
                suffix=suffix,
            ),
            extra_body=extra_body,
        )

        return await self._maybe_overwrite_id(resp, stream)  # type: ignore[no-any-return]

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
        """
        Direct OpenAI chat completion API call.
        """
        if self.download_images:

            async def _localize_image_url(m: OpenAIMessageParam) -> OpenAIMessageParam:
                if isinstance(m.content, list):
                    for c in m.content:
                        if c.type == "image_url" and c.image_url and c.image_url.url and "http" in c.image_url.url:
                            localize_result = await localize_image_content(c.image_url.url)
                            if localize_result is None:
                                raise ValueError(
                                    f"Failed to localize image content from {c.image_url.url[:42]}{'...' if len(c.image_url.url) > 42 else ''}"
                                )
                            content, format = localize_result
                            c.image_url.url = f"data:image/{format};base64,{base64.b64encode(content).decode('utf-8')}"
                # else it's a string and we don't need to modify it
                return m

            messages = [await _localize_image_url(m) for m in messages]

        resp = await self.client.chat.completions.create(
            **await prepare_openai_completion_params(
                model=await self._get_provider_model_id(model),
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
        )

        return await self._maybe_overwrite_id(resp, stream)  # type: ignore[no-any-return]

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Direct OpenAI embeddings API call.
        """
        # Call OpenAI embeddings API with properly typed parameters
        response = await self.client.embeddings.create(
            model=await self._get_provider_model_id(model),
            input=input,
            encoding_format=encoding_format if encoding_format is not None else NOT_GIVEN,
            dimensions=dimensions if dimensions is not None else NOT_GIVEN,
            user=user if user is not None else NOT_GIVEN,
        )

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=model,
            usage=usage,
        )

    async def list_models(self) -> list[Model] | None:
        """
        List available models from the provider's /v1/models endpoint augmented with static embedding model metadata.

        Also, caches the models in self._model_cache for use in check_model_availability().

        :return: A list of Model instances representing available models.
        """
        self._model_cache = {}

        async for m in self.client.models.list():
            if self.allowed_models and m.id not in self.allowed_models:
                logger.info(f"Skipping model {m.id} as it is not in the allowed models list")
                continue
            if metadata := self.embedding_model_metadata.get(m.id):
                # This is an embedding model - augment with metadata
                model = Model(
                    provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                    provider_resource_id=m.id,
                    identifier=m.id,
                    model_type=ModelType.embedding,
                    metadata=metadata,
                )
            else:
                # This is an LLM
                model = Model(
                    provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                    provider_resource_id=m.id,
                    identifier=m.id,
                    model_type=ModelType.llm,
                )
            self._model_cache[m.id] = model

        return list(self._model_cache.values())

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from the provider's /v1/models.

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        if not self._model_cache:
            await self.list_models()

        return model in self._model_cache
