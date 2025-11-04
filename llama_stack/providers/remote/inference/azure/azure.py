# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urljoin

from llama_stack.apis.inference import ChatCompletionRequest
from llama_stack.providers.utils.inference.litellm_openai_mixin import (
    LiteLLMOpenAIMixin,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import AzureConfig


class AzureInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    def __init__(self, config: AzureConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            litellm_provider_name="azure",
            api_key_from_config=config.api_key.get_secret_value(),
            provider_data_api_key_field="azure_api_key",
            openai_compat_api_base=str(config.api_base),
        )
        self.config = config

    # Delegate the client data handling get_api_key method to LiteLLMOpenAIMixin
    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        """
        Get the Azure API base URL.

        Returns the Azure API base URL from the configuration.
        """
        return urljoin(str(self.config.api_base), "/openai/v1")

    async def _get_params(self, request: ChatCompletionRequest) -> dict[str, Any]:
        # Get base parameters from parent
        params = await super()._get_params(request)

        # Add Azure specific parameters
        provider_data = self.get_request_provider_data()
        if provider_data:
            if provider_data.azure_api_key:
                params["api_key"] = provider_data.azure_api_key.get_secret_value()
            if provider_data.azure_api_base:
                params["api_base"] = str(provider_data.azure_api_base)
            if provider_data.azure_api_version:
                params["api_version"] = provider_data.azure_api_version
            if provider_data.azure_api_type:
                params["api_type"] = provider_data.azure_api_type
        else:
            params["api_key"] = self.config.api_key.get_secret_value()
            params["api_base"] = str(self.config.api_base)
            params["api_version"] = self.config.api_version
            params["api_type"] = self.config.api_type

        return params
