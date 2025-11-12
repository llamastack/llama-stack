# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import struct
from collections.abc import AsyncIterator

import litellm

from llama_stack.apis.inference import (
    InferenceProvider,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
)
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper, ProviderModelEntry
from llama_stack.providers.utils.inference.openai_compat import (
    prepare_openai_completion_params,
)

logger = get_logger(name=__name__, category="providers::utils")


class LiteLLMOpenAIMixin(
    ModelRegistryHelper,
    InferenceProvider,
    NeedsRequestProviderData,
):
    # TODO: avoid exposing the litellm specific model names to the user.
    #       potential change: add a prefix param that gets added to the model name
    #                         when calling litellm.
    def __init__(
        self,
        litellm_provider_name: str,
        api_key_from_config: str | None,
        provider_data_api_key_field: str | None = None,
        model_entries: list[ProviderModelEntry] | None = None,
        openai_compat_api_base: str | None = None,
        download_images: bool = False,
        json_schema_strict: bool = True,
    ):
        """
        Initialize the LiteLLMOpenAIMixin.

        :param model_entries: The model entries to register.
        :param api_key_from_config: The API key to use from the config.
        :param provider_data_api_key_field: The field in the provider data that contains the API key (optional).
        :param litellm_provider_name: The name of the provider, used for model lookups.
        :param openai_compat_api_base: The base URL for OpenAI compatibility, or None if not using OpenAI compatibility.
        :param download_images: Whether to download images and convert to base64 for message conversion.
        :param json_schema_strict: Whether to use strict mode for JSON schema validation.
        """
        ModelRegistryHelper.__init__(self, model_entries=model_entries)

        self.litellm_provider_name = litellm_provider_name
        self.api_key_from_config = api_key_from_config
        self.provider_data_api_key_field = provider_data_api_key_field
        self.api_base = openai_compat_api_base
        self.download_images = download_images
        self.json_schema_strict = json_schema_strict

        if openai_compat_api_base:
            self.is_openai_compat = True
        else:
            self.is_openai_compat = False

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_litellm_model_name(self, model_id: str) -> str:
        # users may be using openai/ prefix in their model names. the openai/models.py did this by default.
        # model_id.startswith("openai/") is for backwards compatibility.
        return (
            f"{self.litellm_provider_name}/{model_id}"
            if self.is_openai_compat and not model_id.startswith(self.litellm_provider_name)
            else model_id
        )

    def _add_additional_properties_recursive(self, schema):
        """
        Recursively add additionalProperties: False to all object schemas
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                schema["additionalProperties"] = False

                # Add required field with all property keys if properties exist
                if "properties" in schema and schema["properties"]:
                    schema["required"] = list(schema["properties"].keys())

            if "properties" in schema:
                for prop_schema in schema["properties"].values():
                    self._add_additional_properties_recursive(prop_schema)

            for key in ["anyOf", "allOf", "oneOf"]:
                if key in schema:
                    for sub_schema in schema[key]:
                        self._add_additional_properties_recursive(sub_schema)

            if "not" in schema:
                self._add_additional_properties_recursive(schema["not"])

            # Handle $defs/$ref
            if "$defs" in schema:
                for def_schema in schema["$defs"].values():
                    self._add_additional_properties_recursive(def_schema)

        return schema

    def get_api_key(self) -> str:
        provider_data = self.get_request_provider_data()
        key_field = self.provider_data_api_key_field
        if provider_data and key_field and (api_key := getattr(provider_data, key_field, None)):
            return str(api_key)  # type: ignore[no-any-return]  # getattr returns Any, can't narrow without runtime type inspection

        api_key = self.api_key_from_config
        if not api_key:
            raise ValueError(
                "API key is not set. Please provide a valid API key in the "
                "provider data header, e.g. x-llamastack-provider-data: "
                f'{{"{key_field}": "<API_KEY>"}}, or in the provider config.'
            )
        return api_key

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        if not self.model_store:
            raise ValueError("Model store is not initialized")

        model_obj = await self.model_store.get_model(params.model)
        if model_obj.provider_resource_id is None:
            raise ValueError(f"Model {params.model} has no provider_resource_id")
        provider_resource_id = model_obj.provider_resource_id

        # Convert input to list if it's a string
        input_list = [params.input] if isinstance(params.input, str) else params.input

        # Call litellm embedding function
        # litellm.drop_params = True
        response = litellm.embedding(
            model=self.get_litellm_model_name(provider_resource_id),
            input=input_list,
            api_key=self.get_api_key(),
            api_base=self.api_base,
            dimensions=params.dimensions,
        )

        # Convert response to OpenAI format
        data = b64_encode_openai_embeddings_response(response.data, params.encoding_format)

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response["usage"]["prompt_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=provider_resource_id,
            usage=usage,
        )

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        if not self.model_store:
            raise ValueError("Model store is not initialized")

        model_obj = await self.model_store.get_model(params.model)
        if model_obj.provider_resource_id is None:
            raise ValueError(f"Model {params.model} has no provider_resource_id")
        provider_resource_id = model_obj.provider_resource_id

        request_params = await prepare_openai_completion_params(
            model=self.get_litellm_model_name(provider_resource_id),
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
            api_key=self.get_api_key(),
            api_base=self.api_base,
        )
        # LiteLLM returns compatible type but mypy can't verify external library
        return await litellm.atext_completion(**request_params)  # type: ignore[no-any-return]  # external lib lacks type stubs

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        # Add usage tracking for streaming when telemetry is active
        from llama_stack.core.telemetry.tracing import get_current_span

        stream_options = params.stream_options
        if params.stream and get_current_span() is not None:
            if stream_options is None:
                stream_options = {"include_usage": True}
            elif "include_usage" not in stream_options:
                stream_options = {**stream_options, "include_usage": True}

        if not self.model_store:
            raise ValueError("Model store is not initialized")

        model_obj = await self.model_store.get_model(params.model)
        if model_obj.provider_resource_id is None:
            raise ValueError(f"Model {params.model} has no provider_resource_id")
        provider_resource_id = model_obj.provider_resource_id

        request_params = await prepare_openai_completion_params(
            model=self.get_litellm_model_name(provider_resource_id),
            messages=params.messages,
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
            stream_options=stream_options,
            temperature=params.temperature,
            tool_choice=params.tool_choice,
            tools=params.tools,
            top_logprobs=params.top_logprobs,
            top_p=params.top_p,
            user=params.user,
            api_key=self.get_api_key(),
            api_base=self.api_base,
        )
        # LiteLLM returns compatible type but mypy can't verify external library
        return await litellm.acompletion(**request_params)  # type: ignore[no-any-return]  # external lib lacks type stubs

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available via LiteLLM for the current
        provider (self.litellm_provider_name).

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        if self.litellm_provider_name not in litellm.models_by_provider:
            logger.error(f"Provider {self.litellm_provider_name} is not registered in litellm.")
            return False

        return model in litellm.models_by_provider[self.litellm_provider_name]


def b64_encode_openai_embeddings_response(
    response_data: list[dict], encoding_format: str | None = "float"
) -> list[OpenAIEmbeddingData]:
    """
    Process the OpenAI embeddings response to encode the embeddings in base64 format if specified.
    """
    data = []
    for i, embedding_data in enumerate(response_data):
        if encoding_format == "base64":
            byte_array = bytearray()
            for embedding_value in embedding_data["embedding"]:
                byte_array.extend(struct.pack("f", float(embedding_value)))

            response_embedding = base64.b64encode(byte_array).decode("utf-8")
        else:
            response_embedding = embedding_data["embedding"]
        data.append(
            OpenAIEmbeddingData(
                embedding=response_embedding,
                index=i,
            )
        )
    return data
