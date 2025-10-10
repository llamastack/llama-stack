# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import requests

from llama_stack.apis.inference import ChatCompletionRequest
from llama_stack.apis.models import Model
from llama_stack.apis.models.models import ModelType
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin


class WatsonXInferenceAdapter(LiteLLMOpenAIMixin):
    _model_cache: dict[str, Model] = {}

    def __init__(self, config: WatsonXConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            litellm_provider_name="watsonx",
            api_key_from_config=config.auth_credential.get_secret_value() if config.auth_credential else None,
            provider_data_api_key_field="watsonx_api_key",
        )
        self.available_models = None
        self.config = config

    def get_base_url(self) -> str:
        return self.config.url

    async def _get_params(self, request: ChatCompletionRequest) -> dict[str, Any]:
        # Get base parameters from parent
        params = await super()._get_params(request)

        # Add watsonx.ai specific parameters
        params["project_id"] = self.config.project_id
        params["time_limit"] = self.config.timeout
        return params

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
                embedding_dimension = model_spec["model_limits"]["embedding_dimension"]
                context_length = model_spec["model_limits"]["max_sequence_length"]
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
                # In theory, I guess it is possible that a model could be both an embedding model and a text chat model.
                # In that case, the cache will record the generator Model object, and the list which we return will have
                # both the generator Model object and the text chat Model object.  That's fine because the cache is
                # only used for check_model_availability() anyway.
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
