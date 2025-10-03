# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from typing import Any

import requests

from llama_stack.apis.inference import ChatCompletionRequest
from llama_stack.apis.models import Model
from llama_stack.apis.models.models import ModelType
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin


class WatsonXInferenceAdapter(LiteLLMOpenAIMixin):
    _config: WatsonXConfig
    __provider_id__: str = "watsonx"

    def __init__(self, config: WatsonXConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            litellm_provider_name="watsonx",
            api_key_from_config=config.api_key.get_secret_value(),
            provider_data_api_key_field="watsonx_api_key",
        )
        self.available_models = None
        self.config = config

    # get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        return self.config.url

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()

    async def _get_params(self, request: ChatCompletionRequest) -> dict[str, Any]:
        # Get base parameters from parent
        params = await super()._get_params(request)

        # Add watsonx.ai specific parameters
        params["project_id"] = self.config.project_id
        params["time_limit"] = self.config.timeout
        return params

    async def check_model_availability(self, model):
        return True

    async def list_models(self) -> list[Model] | None:
        models = []
        for model_spec in self._get_model_specs():
            models.append(
                Model(
                    identifier=model_spec["model_id"],
                    provider_resource_id=f"{self.__provider_id__}/{model_spec['model_id']}",
                    provider_id=self.__provider_id__,
                    metadata={},
                    model_type=ModelType.llm,
                )
            )
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


# TO DO: Delete the test main method.
if __name__ == "__main__":
    config = WatsonXConfig(url="https://us-south.ml.cloud.ibm.com", api_key="xxx", project_id="xxx", timeout=60)
    adapter = WatsonXInferenceAdapter(config)
    model_specs = adapter._get_model_specs()
    models = asyncio.run(adapter.list_models())
    for model in models:
        print(model.identifier)
        print(model.provider_resource_id)
        print(model.provider_id)
        print(model.metadata)
        print(model.model_type)
        print("--------------------------------")
