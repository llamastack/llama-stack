# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import openai
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
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params

logger = get_logger(name=__name__, category="providers::utils")


class OpenAIMixin(ABC):
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
        if guided_choice is not None:
            logger.warning("guided_choice is not supported by the OpenAI API. Ignoring.")
        if prompt_logprobs is not None:
            logger.warning("prompt_logprobs is not supported by the OpenAI API. Ignoring.")

        # TODO: fix openai_completion to return type compatible with OpenAI's API response
        return await self.client.completions.create(  # type: ignore[no-any-return]
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
            )
        )

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
        # Type ignore because return types are compatible
        return await self.client.chat.completions.create(  # type: ignore[no-any-return]
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
            model=response.model,
            usage=usage,
        )

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from OpenAI.

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        try:
            # Direct model lookup - returns model or raises NotFoundError
            await self.client.models.retrieve(model)
            return True
        except openai.NotFoundError:
            # Model doesn't exist - this is expected for unavailable models
            pass
        except Exception as e:
            # All other errors (auth, rate limit, network, etc.)
            logger.warning(f"Failed to check model availability for {model}: {e}")

        return False
