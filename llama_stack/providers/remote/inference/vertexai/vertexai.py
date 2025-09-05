# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import SecretStr

from llama_stack.apis.inference import ChatCompletionRequest
from llama_stack.providers.utils.inference.litellm_openai_mixin import (
    LiteLLMOpenAIMixin,
)

from .config import VertexAIConfig
from .models import MODEL_ENTRIES


class VertexAIInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: VertexAIConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            litellm_provider_name="vertex_ai",
            api_key_from_config=None,  # Vertex AI uses ADC, not API keys
            provider_data_api_key_field="vertex_project",  # Use project for validation
        )
        self.config = config

    def get_api_key(self) -> SecretStr:
        # Vertex AI doesn't use API keys, it uses Application Default Credentials
        # Return empty string to let litellm handle authentication via ADC
        return SecretStr("")

    async def _get_params(self, request: ChatCompletionRequest) -> dict[str, Any]:
        # Get base parameters from parent
        params = await super()._get_params(request)

        # Add Vertex AI specific parameters
        provider_data = self.get_request_provider_data()
        if provider_data:
            if getattr(provider_data, "vertex_project", None):
                params["vertex_project"] = provider_data.vertex_project
            if getattr(provider_data, "vertex_location", None):
                params["vertex_location"] = provider_data.vertex_location
        else:
            params["vertex_project"] = self.config.project
            params["vertex_location"] = self.config.location

        # Remove api_key since Vertex AI uses ADC
        params.pop("api_key", None)

        return params
