# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import google.auth.transport.requests
from google.auth import default

from llama_stack.apis.inference import ChatCompletionRequest
from llama_stack.providers.utils.inference.litellm_openai_mixin import (
    LiteLLMOpenAIMixin,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import VertexAIConfig
from .models import MODEL_ENTRIES


class VertexAIInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    def __init__(self, config: VertexAIConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            litellm_provider_name="vertex_ai",
            api_key_from_config=None,  # Vertex AI uses ADC, not API keys
            provider_data_api_key_field="vertex_project",  # Use project for validation
        )
        self.config = config

    def get_api_key(self) -> str:
        """
        Get an access token for Vertex AI using Application Default Credentials.

        Vertex AI uses ADC instead of API keys. This method obtains an access token
        from the default credentials and returns it for use with the OpenAI-compatible client.
        """
        try:
            # Get default credentials - will read from GOOGLE_APPLICATION_CREDENTIALS
            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
            return str(credentials.token)
        except Exception:
            # If we can't get credentials, return empty string to let LiteLLM handle it
            # This allows the LiteLLM mixin to work with ADC directly
            return ""

    def get_base_url(self) -> str:
        """
        Get the Vertex AI OpenAI-compatible API base URL.

        Returns the Vertex AI OpenAI-compatible endpoint URL.
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/start/openai
        """
        return f"https://{self.config.location}-aiplatform.googleapis.com/v1/projects/{self.config.project}/locations/{self.config.location}/endpoints/openapi"

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
