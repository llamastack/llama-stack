# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

import litellm
import requests

from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionUsage,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.apis.models import Model
from llama_stack.apis.models.models import ModelType
from llama_stack.core.telemetry.tracing import get_current_span
from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params

logger = get_logger(name=__name__, category="providers::remote::watsonx")


class WatsonXInferenceAdapter(LiteLLMOpenAIMixin):
    _model_cache: dict[str, Model] = {}

    provider_data_api_key_field: str = "watsonx_api_key"

    def __init__(self, config: WatsonXConfig):
        self.available_models = None
        self.config = config
        api_key = config.auth_credential.get_secret_value() if config.auth_credential else None
        LiteLLMOpenAIMixin.__init__(
            self,
            litellm_provider_name="watsonx",
            api_key_from_config=api_key,
            provider_data_api_key_field="watsonx_api_key",
            openai_compat_api_base=self.get_base_url(),
        )

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """
        Override parent method to add timeout and inject usage object when missing.
        This works around a LiteLLM defect where usage block is sometimes dropped.
        """

        # Add usage tracking for streaming when telemetry is active
        stream_options = params.stream_options
        if params.stream and get_current_span() is not None:
            if stream_options is None:
                stream_options = {"include_usage": True}
            elif "include_usage" not in stream_options:
                stream_options = {**stream_options, "include_usage": True}

        model_obj = await self.model_store.get_model(params.model)

        request_params = await prepare_openai_completion_params(
            model=self.get_litellm_model_name(model_obj.provider_resource_id),
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
            # These are watsonx-specific parameters
            timeout=self.config.timeout,
            project_id=self.config.project_id,
        )

        result = await litellm.acompletion(**request_params)

        # If not streaming, check and inject usage if missing
        if not params.stream:
            # Use getattr to safely handle cases where usage attribute might not exist
            if getattr(result, "usage", None) is None:
                # Create usage object with zeros
                usage_obj = OpenAIChatCompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                # Use model_copy to create a new response with the usage injected
                result = result.model_copy(update={"usage": usage_obj})
            return result

        # For streaming, wrap the iterator to normalize chunks
        return self._normalize_stream(result)

    def _normalize_chunk(self, chunk: OpenAIChatCompletionChunk) -> OpenAIChatCompletionChunk:
        """
        Normalize a chunk to ensure it has all expected attributes.
        This works around LiteLLM not always including all expected attributes.
        """
        # Ensure chunk has usage attribute with zeros if missing
        if not hasattr(chunk, "usage") or chunk.usage is None:
            usage_obj = OpenAIChatCompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            chunk = chunk.model_copy(update={"usage": usage_obj})

        # Ensure all delta objects in choices have expected attributes
        if hasattr(chunk, "choices") and chunk.choices:
            normalized_choices = []
            for choice in chunk.choices:
                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta
                    # Build update dict for missing attributes
                    delta_updates = {}
                    if not hasattr(delta, "refusal"):
                        delta_updates["refusal"] = None
                    if not hasattr(delta, "reasoning_content"):
                        delta_updates["reasoning_content"] = None

                    # If we need to update delta, create a new choice with updated delta
                    if delta_updates:
                        new_delta = delta.model_copy(update=delta_updates)
                        new_choice = choice.model_copy(update={"delta": new_delta})
                        normalized_choices.append(new_choice)
                    else:
                        normalized_choices.append(choice)
                else:
                    normalized_choices.append(choice)

            # If we modified any choices, create a new chunk with updated choices
            if any(normalized_choices[i] is not chunk.choices[i] for i in range(len(chunk.choices))):
                chunk = chunk.model_copy(update={"choices": normalized_choices})

        return chunk

    async def _normalize_stream(
        self, stream: AsyncIterator[OpenAIChatCompletionChunk]
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        """
        Normalize all chunks in the stream to ensure they have expected attributes.
        This works around LiteLLM sometimes not including expected attributes.
        """
        try:
            async for chunk in stream:
                # Normalize and yield each chunk immediately
                yield self._normalize_chunk(chunk)
        except Exception as e:
            logger.error(f"Error normalizing stream: {e}", exc_info=True)
            raise

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        """
        Override parent method to add watsonx-specific parameters.
        """
        from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params

        model_obj = await self.model_store.get_model(params.model)

        request_params = await prepare_openai_completion_params(
            model=self.get_litellm_model_name(model_obj.provider_resource_id),
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
            # These are watsonx-specific parameters
            timeout=self.config.timeout,
            project_id=self.config.project_id,
        )
        return await litellm.atext_completion(**request_params)

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """
        Override parent method to add watsonx-specific parameters.
        """
        model_obj = await self.model_store.get_model(params.model)

        # Convert input to list if it's a string
        input_list = [params.input] if isinstance(params.input, str) else params.input

        # Call litellm embedding function with watsonx-specific parameters
        response = litellm.embedding(
            model=self.get_litellm_model_name(model_obj.provider_resource_id),
            input=input_list,
            api_key=self.get_api_key(),
            api_base=self.api_base,
            dimensions=params.dimensions,
            # These are watsonx-specific parameters
            timeout=self.config.timeout,
            project_id=self.config.project_id,
        )

        # Convert response to OpenAI format
        from llama_stack.apis.inference import OpenAIEmbeddingUsage
        from llama_stack.providers.utils.inference.litellm_openai_mixin import b64_encode_openai_embeddings_response

        data = b64_encode_openai_embeddings_response(response.data, params.encoding_format)

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response["usage"]["prompt_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=model_obj.provider_resource_id,
            usage=usage,
        )

    def get_base_url(self) -> str:
        return self.config.url

    # Copied from OpenAIMixin
    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from the provider's /v1/models.

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        if not self._model_cache:
            await self.list_models()
        return model in self._model_cache

    async def list_models(self) -> list[Model] | None:
        self._model_cache = {}
        models = []
        for model_spec in self._get_model_specs():
            functions = [f["id"] for f in model_spec.get("functions", [])]
            # Format: {"embedding_dimension": 1536, "context_length": 8192}

            # Example of an embedding model:
            # {'model_id': 'ibm/granite-embedding-278m-multilingual',
            # 'label': 'granite-embedding-278m-multilingual',
            # 'model_limits': {'max_sequence_length': 512, 'embedding_dimension': 768},
            # ...
            provider_resource_id = f"{self.__provider_id__}/{model_spec['model_id']}"
            if "embedding" in functions:
                embedding_dimension = model_spec.get("model_limits", {}).get("embedding_dimension", 0)
                context_length = model_spec.get("model_limits", {}).get("max_sequence_length", 0)
                embedding_metadata = {
                    "embedding_dimension": embedding_dimension,
                    "context_length": context_length,
                }
                model = Model(
                    identifier=model_spec["model_id"],
                    provider_resource_id=provider_resource_id,
                    provider_id=self.__provider_id__,
                    metadata=embedding_metadata,
                    model_type=ModelType.embedding,
                )
                self._model_cache[provider_resource_id] = model
                models.append(model)
            if "text_chat" in functions:
                model = Model(
                    identifier=model_spec["model_id"],
                    provider_resource_id=provider_resource_id,
                    provider_id=self.__provider_id__,
                    metadata={},
                    model_type=ModelType.llm,
                )
                self._model_cache[provider_resource_id] = model
                models.append(model)
        return models

    # LiteLLM provides methods to list models for many providers, but not for watsonx.ai.
    # So we need to implement our own method to list models by calling the watsonx.ai API.
    def _get_model_specs(self) -> list[dict[str, Any]]:
        """
        Retrieves foundation model specifications from the watsonx.ai API.
        """
        url = f"{self.config.url}/ml/v1/foundation_model_specs?version=2023-10-25"
        headers = {
            # Note that there is no authorization header.  Listing models does not require authentication.
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers)

        # --- Process the Response ---
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # If the request is successful, parse and return the JSON response.
        # The response should contain a list of model specifications
        response_data = response.json()
        if "resources" not in response_data:
            raise ValueError("Resources not found in response")
        return response_data["resources"]
