# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable

import google.auth.transport.requests
from google.auth import default

from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import VertexAIConfig


class VertexAIInferenceAdapter(OpenAIMixin):
    config: VertexAIConfig

    provider_data_api_key_field: str = "vertex_project"

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
            # If we can't get credentials, return empty string to let the env work with ADC directly
            return ""

    def get_base_url(self) -> str:
        """
        Get the Vertex AI OpenAI-compatible API base URL.

        Returns the Vertex AI OpenAI-compatible endpoint URL.
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/start/openai
        """
        return f"https://{self.config.location}-aiplatform.googleapis.com/v1/projects/{self.config.project}/locations/{self.config.location}/endpoints/openapi"

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        VertexAI doesn't currently offer a way to query a list of available models from Google's Model Garden
        For now we return a hardcoded version of the available models

        :return: An iterable of model IDs
        """
        return ["vertexai/gemini-2.0-flash", "vertexai/gemini-2.5-flash", "vertexai/gemini-2.5-pro"]
