# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from urllib.parse import urljoin

from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import AzureConfig


class AzureInferenceAdapter(OpenAIMixin):
    config: AzureConfig

    provider_data_api_key_field: str = "azure_api_key"

    def get_base_url(self) -> str:
        """
        Get the Azure API base URL.

        Returns the Azure API base URL from the configuration.
        """
        return urljoin(str(self.config.api_base), "/openai/v1")
