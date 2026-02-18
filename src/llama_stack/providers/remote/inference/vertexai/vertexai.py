# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any, cast

from google.genai import Client
from google.genai import types as genai_types
from google.oauth2.credentials import Credentials

from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.vertexai import converters
from llama_stack.providers.remote.inference.vertexai.config import (
    VertexAIConfig,
    VertexAIProviderDataValidator,
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack_api import (
    Inference,
    Model,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankResponse,
)
from llama_stack_api.inference import RerankRequest

logger = get_logger(__name__, category="inference")


class VertexAIInferenceAdapter(ModelRegistryHelper, NeedsRequestProviderData, Inference):
    config: VertexAIConfig

    def __init__(self, config: VertexAIConfig) -> None:
        ModelRegistryHelper.__init__(
            self,
            model_entries=[],
            allowed_models=config.allowed_models,
        )
        self.config = config
        self._default_client: Client | None = None

    async def initialize(self) -> None:
        if self.config.auth_credential is not None:
            logger.warning(
                "VertexAIConfig.auth_credential is deprecated and has no effect. "
                "Use access_token or Application Default Credentials (ADC) instead.",
            )
        try:
            access_token = self.config.access_token.get_secret_value() if self.config.access_token else None
            self._default_client = self._create_client(
                project=self.config.project,
                location=self.config.location,
                access_token=access_token,
            )
            logger.info(
                "VertexAI client initialized for project=%s location=%s",
                self.config.project,
                self.config.location,
            )
        except Exception:
            logger.warning(
                "Failed to initialize default VertexAI client. Requests will require explicit credentials.",
                exc_info=True,
            )
            self._default_client = None

    async def shutdown(self) -> None:
        pass

    @staticmethod
    @lru_cache(maxsize=4)
    def _create_adc_client(project: str, location: str) -> Client:
        """Create a cached client using Application Default Credentials."""
        return Client(vertexai=True, project=project, location=location)

    @staticmethod
    def _create_client(project: str, location: str, *, access_token: str | None = None) -> Client:
        """Create a VertexAI client.

        When *access_token* is provided the client is **not** cached because
        OAuth tokens are short-lived (~1 h) and caching would cause silent
        auth failures after expiry.  ADC clients are cached via
        ``_create_adc_client``.
        """
        if access_token:
            credentials = Credentials(token=access_token)
            return Client(vertexai=True, project=project, location=location, credentials=credentials)
        return VertexAIInferenceAdapter._create_adc_client(project, location)

    async def check_model_availability(self, model: str) -> bool:
        """Check whether *model* is served by the configured VertexAI project.

        Falls back to accepting the model when the API is unreachable so
        that configured models are not rejected during offline startup.
        """
        try:
            available = await self.list_provider_model_ids()
        except Exception:
            logger.warning(
                "Failed to list VertexAI models for availability check; accepting model '%s' without validation.",
                model,
                exc_info=True,
            )
            return True

        if model in available:
            return True
        # Accept with or without the "google/" vendor prefix.
        bare = model.removeprefix("google/")
        prefixed = f"google/{bare}"
        return bare in available or prefixed in available

    @staticmethod
    def _supports_generate_content(model: Any) -> bool:
        actions = getattr(model, "supported_actions", None)
        if actions is None and isinstance(model, dict):
            actions = model.get("supported_actions")
        actions = actions or []
        if not actions:
            return True
        return "generateContent" in actions

    def _get_request_provider_overrides(self) -> VertexAIProviderDataValidator | None:
        provider_data = self.get_request_provider_data()
        if provider_data is None:
            return None

        if isinstance(provider_data, VertexAIProviderDataValidator):
            return provider_data

        if isinstance(provider_data, dict):
            return VertexAIProviderDataValidator(**provider_data)

        try:
            return VertexAIProviderDataValidator.model_validate(provider_data)
        except Exception:
            logger.warning("Failed to parse VertexAI provider data, falling back to config defaults", exc_info=True)
            return None

    def _get_client(self) -> Client:
        overrides = self._get_request_provider_overrides()
        if overrides is not None:
            project = overrides.vertex_project or self.config.project
            location = overrides.vertex_location or self.config.location
            if overrides.vertex_access_token:
                return self._create_client(
                    project=project, location=location, access_token=overrides.vertex_access_token
                )
            if overrides.vertex_project or overrides.vertex_location:
                access_token = self.config.access_token.get_secret_value() if self.config.access_token else None
                return self._create_client(project=project, location=location, access_token=access_token)

        if self._default_client is None:
            raise RuntimeError("No VertexAI client available. Configure ADC or provide an access token.")
        return self._default_client

    async def _get_provider_model_id(self, model: str) -> str:
        provider_model_id = self.get_provider_model_id(model)
        if provider_model_id:
            return provider_model_id

        if self.model_store is not None and await self.model_store.has_model(model):  # type: ignore[attr-defined]
            model_obj: Model = await self.model_store.get_model(model)
            if model_obj.provider_resource_id is None:
                raise ValueError(f"Model {model} has no provider_resource_id")
            return model_obj.provider_resource_id

        return model

    async def list_provider_model_ids(self) -> list[str]:
        client = self._get_client()
        config = genai_types.ListModelsConfig(query_base=True)
        result: list[str] = []

        async for model in await client.aio.models.list(config=config):
            if not self._supports_generate_content(model):
                continue

            name = getattr(model, "name", "") or ""
            model_id = name.removeprefix("models/")
            if not model_id:
                continue
            if not model_id.startswith("google/"):
                model_id = f"google/{model_id}"
            result.append(model_id)

        return list(dict.fromkeys(result))

    async def list_models(self) -> list[Model] | None:
        """List models available from the configured VertexAI project.

        Queries the Gemini API via ``list_provider_model_ids()`` and constructs
        ``Model`` objects, respecting ``allowed_models`` when configured.
        """
        try:
            provider_model_ids = await self.list_provider_model_ids()
        except Exception:
            logger.error(
                "%s.list_provider_model_ids() failed",
                self.__class__.__name__,
                exc_info=True,
            )
            raise

        models: list[Model] = []
        for provider_model_id in provider_model_ids:
            if self.config.allowed_models is not None and not self._is_model_allowed(provider_model_id):
                continue
            models.append(
                Model(
                    provider_id=self.__provider_id__,
                    provider_resource_id=provider_model_id,
                    identifier=provider_model_id,
                    model_type=ModelType.llm,
                )
            )

        logger.info("%s.list_models() returned %d model(s)", self.__class__.__name__, len(models))
        return models

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

    def _build_tool_config(self, tool_choice: str | dict[str, Any] | None) -> genai_types.ToolConfig | None:
        if tool_choice is None or tool_choice == "auto":
            return None

        if tool_choice == "none":
            return self._make_tool_config(mode="NONE")

        if tool_choice == "required":
            return self._make_tool_config(mode="ANY")

        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("function", {}).get("name")
            if function_name:
                return self._make_tool_config(mode="ANY", allowed_function_names=[function_name])

        return None

    @staticmethod
    def _make_tool_config(
        *,
        mode: str,
        allowed_function_names: list[str] | None = None,
    ) -> genai_types.ToolConfig:
        function_calling_kwargs: dict[str, Any] = {"mode": cast(Any, mode)}
        if allowed_function_names:
            function_calling_kwargs["allowed_function_names"] = allowed_function_names
        function_calling = genai_types.FunctionCallingConfig(**function_calling_kwargs)
        return genai_types.ToolConfig(function_calling_config=function_calling)

    @staticmethod
    def _collect_sampling_params(params: OpenAIChatCompletionRequestWithExtraBody) -> dict[str, Any]:
        """Collect sampling-related config kwargs from the request."""
        kwargs: dict[str, Any] = {}
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.n is not None:
            kwargs["candidate_count"] = params.n

        max_tokens = params.max_completion_tokens if params.max_completion_tokens is not None else params.max_tokens
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        if params.stop is not None:
            kwargs["stop_sequences"] = [params.stop] if isinstance(params.stop, str) else list(params.stop)

        if params.response_format is not None:
            kwargs.update(converters.convert_response_format(params.response_format.model_dump(exclude_none=True)))

        return kwargs

    def _build_generation_config(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
        *,
        system_instruction: str | None,
        tools_input: list[dict[str, Any]] | None,
    ) -> genai_types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` from the OpenAI request parameters."""
        config_kwargs = self._collect_sampling_params(params)

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools_input:
            config_kwargs["tools"] = [genai_types.Tool(**tool) for tool in tools_input]

        tool_config = self._build_tool_config(params.tool_choice)
        if tool_config is not None:
            config_kwargs["tool_config"] = tool_config

        if params.model_extra:
            config_kwargs.update(params.model_extra)

        return genai_types.GenerateContentConfig(**config_kwargs)

    async def _stream_chat_completion(
        self,
        client: Client,
        provider_model_id: str,
        contents: Any,
        config: genai_types.GenerateContentConfig,
        model: str,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        stream = await client.aio.models.generate_content_stream(
            model=provider_model_id,
            contents=contents,
            config=config,
        )
        completion_id = converters.generate_completion_id()

        async def _iter() -> AsyncIterator[OpenAIChatCompletionChunk]:
            is_first_chunk = True
            async for chunk in stream:
                yield converters.convert_gemini_stream_chunk_to_openai(
                    chunk=chunk,
                    model=model,
                    completion_id=completion_id,
                    is_first_chunk=is_first_chunk,
                )
                is_first_chunk = False

        return _iter()

    def _is_model_allowed(self, provider_model_id: str) -> bool:
        # Check with and without the "google/" prefix so users can specify
        # either form in allowed_models or in the request.
        if self.config.allowed_models is None:
            return True
        bare = provider_model_id.removeprefix("google/")
        prefixed = f"google/{bare}"
        return bare in self.config.allowed_models or prefixed in self.config.allowed_models

    def _validate_model_allowed(self, provider_model_id: str) -> None:
        if not self._is_model_allowed(provider_model_id):
            raise ValueError(
                f"Model '{provider_model_id}' is not in the allowed models list. "
                f"Allowed models: {self.config.allowed_models}"
            )

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)
        gemini_model_id = converters.convert_model_name(provider_model_id)
        client = self._get_client()

        system_instruction, contents = converters.convert_openai_messages_to_gemini(params.messages)
        tools_input = converters.convert_openai_tools_to_gemini(params.tools)
        config = self._build_generation_config(params, system_instruction=system_instruction, tools_input=tools_input)

        request_contents = cast(Any, contents)

        if params.stream:
            return await self._stream_chat_completion(client, gemini_model_id, request_contents, config, params.model)

        response = await client.aio.models.generate_content(
            model=gemini_model_id,
            contents=request_contents,
            config=config,
        )
        return converters.convert_gemini_response_to_openai(response=response, model=params.model)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        _ = params
        raise NotImplementedError("VertexAI does not support text completions")

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        _ = params
        raise NotImplementedError("VertexAI embeddings not yet implemented")

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        _ = request
        raise NotImplementedError("VertexAI rerank not yet implemented")
