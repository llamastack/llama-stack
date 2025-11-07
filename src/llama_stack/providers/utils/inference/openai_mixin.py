# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from llama_stack.apis.inference import (
    Model,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    OpenAIMessageParam,
)
from llama_stack.apis.models import ModelType
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params
from llama_stack.providers.utils.inference.prompt_adapter import localize_image_content

logger = get_logger(name=__name__, category="providers::utils")


class OpenAIMixin(NeedsRequestProviderData, ABC, BaseModel):
    """
    Mixin class that provides OpenAI-specific functionality for inference providers.
    This class handles direct OpenAI API calls using the AsyncOpenAI client.

    This is an abstract base class that requires child classes to implement:
    - get_base_url(): Method to retrieve the OpenAI-compatible API base URL

    The behavior of this class can be customized by child classes in the following ways:
    - overwrite_completion_id: If True, overwrites the 'id' field in OpenAI responses
    - download_images: If True, downloads images and converts to base64 for providers that require it
    - embedding_model_metadata: A dictionary mapping model IDs to their embedding metadata
    - construct_model_from_identifier: Method to construct a Model instance corresponding to the given identifier
    - provider_data_api_key_field: Optional field name in provider data to look for API key
    - list_provider_model_ids: Method to list available models from the provider
    - get_extra_client_params: Method to provide extra parameters to the AsyncOpenAI client

    Expected Dependencies:
    - self.model_store: Injected by the Llama Stack distribution system at runtime.
      This provides model registry functionality for looking up registered models.
      The model_store is set in routing_tables/common.py during provider initialization.
    """

    # Allow extra fields so the routing infra can inject model_store, __provider_id__, etc.
    model_config = ConfigDict(extra="allow")

    config: RemoteInferenceProviderConfig

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

    # Optional field name in provider data to look for API key, which takes precedence
    provider_data_api_key_field: str | None = None

    def get_api_key(self) -> str | None:
        """
        Get the API key.

        :return: The API key as a string, or None if not set
        """
        if self.config.auth_credential is None:
            return None
        return self.config.auth_credential.get_secret_value()

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

    def construct_model_from_identifier(self, identifier: str) -> Model:
        """
        Construct a Model instance corresponding to the given identifier

        Child classes can override this to customize model typing/metadata.

        :param identifier: The provider's model identifier
        :return: A Model instance
        """
        if metadata := self.embedding_model_metadata.get(identifier):
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.embedding,
                metadata=metadata,
            )
        return Model(
            provider_id=self.__provider_id__,  # type: ignore[attr-defined]
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.llm,
        )

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        List available models from the provider.

        Child classes can override this method to provide a custom implementation
        for listing models. The default implementation uses the AsyncOpenAI client
        to list models from the OpenAI-compatible endpoint.

        :return: An iterable of model IDs or None if not implemented
        """
        client = self.client
        async with client:
            model_ids = [m.id async for m in client.models.list()]
        return model_ids

    async def initialize(self) -> None:
        """
        Initialize the OpenAI mixin.

        This method provides a default implementation that does nothing.
        Subclasses can override this method to perform initialization tasks
        such as setting up clients, validating configurations, etc.
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the OpenAI mixin.

        This method provides a default implementation that does nothing.
        Subclasses can override this method to perform cleanup tasks
        such as closing connections, releasing resources, etc.
        """
        pass

    @property
    def client(self) -> AsyncOpenAI:
        """
        Get an AsyncOpenAI client instance.

        Uses the abstract methods get_api_key() and get_base_url() which must be
        implemented by child classes.

        Users can also provide the API key via the provider data header, which
        is used instead of any config API key.
        """

        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            message = "API key not provided."
            if self.provider_data_api_key_field:
                message += f' Please provide a valid API key in the provider data header, e.g. x-llamastack-provider-data: {{"{self.provider_data_api_key_field}": "<API_KEY>"}}.'
            raise ValueError(message)

        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.get_base_url(),
            **self.get_extra_client_params(),
        )

    def _get_api_key_from_config_or_provider_data(self) -> str | None:
        api_key = self.get_api_key()

        if self.provider_data_api_key_field:
            provider_data = self.get_request_provider_data()
            if provider_data and getattr(provider_data, self.provider_data_api_key_field, None):
                api_key = getattr(provider_data, self.provider_data_api_key_field)

        return api_key

    async def _get_provider_model_id(self, model: str) -> str:
        """
        Get the provider-specific model ID from the model store.

        This is a utility method that looks up the registered model and returns
        the provider_resource_id that should be used for actual API calls.

        :param model: The registered model name/identifier
        :return: The provider-specific model ID (e.g., "gpt-4")
        """
        # self.model_store is injected by the distribution system at runtime
        if not await self.model_store.has_model(model):  # type: ignore[attr-defined]
            return model

        # Look up the registered model to get the provider-specific model ID
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
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        """
        Direct OpenAI completion API call.
        """
        # TODO: fix openai_completion to return type compatible with OpenAI's API response
        completion_kwargs = await prepare_openai_completion_params(
            model=await self._get_provider_model_id(params.model),
            prompt=params.prompt,
            best_of=params.best_of,
            echo=params.echo,
            frequency_penalty=params.frequency_penalty,
            logit_bias=params.logit_bias,
            logprobs=params.logprobs,
            max_tokens=params.max_tokens,
            n=params.n,
            presence_penalty=params.presence_penalty,
            seed=params.seed,
            stop=params.stop,
            stream=params.stream,
            stream_options=params.stream_options,
            temperature=params.temperature,
            top_p=params.top_p,
            user=params.user,
            suffix=params.suffix,
        )
        if extra_body := params.model_extra:
            completion_kwargs["extra_body"] = extra_body
        resp = await self.client.completions.create(**completion_kwargs)

        return await self._maybe_overwrite_id(resp, params.stream)  # type: ignore[no-any-return]

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """
        Direct OpenAI chat completion API call.
        """
        messages = params.messages

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

        request_params = await prepare_openai_completion_params(
            model=await self._get_provider_model_id(params.model),
            messages=messages,
            frequency_penalty=params.frequency_penalty,
            function_call=params.function_call,
            functions=params.functions,
            logit_bias=params.logit_bias,
            logprobs=params.logprobs,
            max_completion_tokens=params.max_completion_tokens,
            max_tokens=params.max_tokens,
            n=params.n,
            parallel_tool_calls=params.parallel_tool_calls,
            presence_penalty=params.presence_penalty,
            response_format=params.response_format,
            seed=params.seed,
            stop=params.stop,
            stream=params.stream,
            stream_options=params.stream_options,
            temperature=params.temperature,
            tool_choice=params.tool_choice,
            tools=params.tools,
            top_logprobs=params.top_logprobs,
            top_p=params.top_p,
            user=params.user,
        )

        if extra_body := params.model_extra:
            request_params["extra_body"] = extra_body
        resp = await self.client.chat.completions.create(**request_params)

        return await self._maybe_overwrite_id(resp, params.stream)  # type: ignore[no-any-return]

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """
        Direct OpenAI embeddings API call.
        """
        # Build request params conditionally to avoid NotGiven/Omit type mismatch
        # The OpenAI SDK uses Omit in signatures but NOT_GIVEN has type NotGiven
        request_params: dict[str, Any] = {
            "model": await self._get_provider_model_id(params.model),
            "input": params.input,
        }
        if params.encoding_format is not None:
            request_params["encoding_format"] = params.encoding_format
        if params.dimensions is not None:
            request_params["dimensions"] = params.dimensions
        if params.user is not None:
            request_params["user"] = params.user
        if params.model_extra:
            request_params["extra_body"] = params.model_extra

        response = await self.client.embeddings.create(**request_params)

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
            model=params.model,
            usage=usage,
        )

    ###
    # ModelsProtocolPrivate implementation - provide model management functionality
    #
    #  async def register_model(self, model: Model) -> Model: ...
    #  async def unregister_model(self, model_id: str) -> None: ...
    #
    #  async def list_models(self) -> list[Model] | None: ...
    #  async def should_refresh_models(self) -> bool: ...
    ##

    async def register_model(self, model: Model) -> Model:
        if not await self.check_model_availability(model.provider_model_id):
            raise ValueError(f"Model {model.provider_model_id} is not available from provider {self.__provider_id__}")  # type: ignore[attr-defined]
        return model

    async def unregister_model(self, model_id: str) -> None:
        return None

    async def list_models(self) -> list[Model] | None:
        """
        List available models from the provider's /v1/models endpoint augmented with static embedding model metadata.

        Also, caches the models in self._model_cache for use in check_model_availability().

        :return: A list of Model instances representing available models.
        """
        self._model_cache = {}

        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            logger.debug(f"{self.__class__.__name__}.list_provider_model_ids() disabled because API key not provided")
            return None

        try:
            iterable = await self.list_provider_model_ids()
        except Exception as e:
            logger.error(f"{self.__class__.__name__}.list_provider_model_ids() failed with: {e}")
            raise
        if not hasattr(iterable, "__iter__"):
            raise TypeError(
                f"Failed to list models: {self.__class__.__name__}.list_provider_model_ids() must return an iterable of "
                f"strings, but returned {type(iterable).__name__}"
            )

        provider_models_ids = list(iterable)
        logger.info(f"{self.__class__.__name__}.list_provider_model_ids() returned {len(provider_models_ids)} models")

        for provider_model_id in provider_models_ids:
            if not isinstance(provider_model_id, str):
                raise ValueError(f"Model ID {provider_model_id} from list_provider_model_ids() is not a string")
            if self.config.allowed_models is not None and provider_model_id not in self.config.allowed_models:
                logger.info(f"Skipping model {provider_model_id} as it is not in the allowed models list")
                continue
            model = self.construct_model_from_identifier(provider_model_id)
            self._model_cache[provider_model_id] = model

        return list(self._model_cache.values())

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from the provider's /v1/models or pre-registered.

        :param model: The model identifier to check.
        :return: True if the model is available dynamically or pre-registered, False otherwise.
        """
        # First check if the model is pre-registered in the model store
        if hasattr(self, "model_store") and self.model_store:
            qualified_model = f"{self.__provider_id__}/{model}"  # type: ignore[attr-defined]
            if await self.model_store.has_model(qualified_model):
                return True

        # Then check the provider's dynamic model cache
        if not self._model_cache:
            await self.list_models()
        return model in self._model_cache

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

    #
    # The model_dump implementations are to avoid serializing the extra fields,
    # e.g. model_store, which are not pydantic.
    #

    def _filter_fields(self, **kwargs):
        """Helper to exclude extra fields from serialization."""
        # Exclude any extra fields stored in __pydantic_extra__
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            exclude = kwargs.get("exclude", set())
            if not isinstance(exclude, set):
                exclude = set(exclude) if exclude else set()
            exclude.update(self.__pydantic_extra__.keys())
            kwargs["exclude"] = exclude
        return kwargs

    def model_dump(self, **kwargs):
        """Override to exclude extra fields from serialization."""
        kwargs = self._filter_fields(**kwargs)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        """Override to exclude extra fields from JSON serialization."""
        kwargs = self._filter_fields(**kwargs)
        return super().model_dump_json(**kwargs)
