# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable

import google.auth.transport.requests
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError, RefreshError, TransportError

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
        except DefaultCredentialsError as e:
            raise ValueError(
                "Vertex AI authentication failed: No credentials found. "
                "Please configure Application Default Credentials (ADC) by setting the GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable to point to a service account JSON file, or run 'gcloud auth application-default login' "
                "to use your user credentials. See https://cloud.google.com/docs/authentication/application-default-credentials"
            ) from e
        except RefreshError as e:
            raise ValueError(
                "Vertex AI authentication failed: Token refresh failed. "
                "This may indicate that your service account credentials are invalid, expired, or lack necessary permissions. "
                "Please verify that your service account has the 'Vertex AI User' role and that the credentials have not been revoked. "
                "If using a service account key file, ensure it has not expired."
            ) from e
        except TransportError as e:
            raise ValueError(
                "Vertex AI authentication failed: Network connectivity issue. "
                "Unable to reach Google's authentication servers. "
                "Please check your network connection, firewall settings, and proxy configuration. "
                "Ensure that your environment can reach https://oauth2.googleapis.com and https://www.googleapis.com"
            ) from e
        except GoogleAuthError as e:
            raise ValueError(f"Vertex AI authentication failed: {e}") from e

    def get_base_url(self) -> str:
        """
        Get the Vertex AI OpenAI-compatible API base URL.

        Returns the Vertex AI OpenAI-compatible endpoint URL.
        Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai
        """
        if not self.config.location or self.config.location == "global":
            return f"https://aiplatform.googleapis.com/v1/projects/{self.config.project}/locations/global/endpoints/openapi"
        else:
            return f"https://{self.config.location}-aiplatform.googleapis.com/v1/projects/{self.config.project}/locations/{self.config.location}/endpoints/openapi"

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        VertexAI doesn't currently offer a way to query a list of available models from Google's Model Garden
        For now we return a hardcoded version of the available models

        :return: An iterable of model IDs
        """
        return ["google/gemini-2.0-flash", "google/gemini-2.5-flash", "google/gemini-2.5-pro"]
