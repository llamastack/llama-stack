# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import SambaNovaImplConfig


class SambaNovaInferenceAdapter(OpenAIMixin):
    config: SambaNovaImplConfig

    provider_data_api_key_field: str = "sambanova_api_key"
    download_images: bool = True  # SambaNova does not support image downloads server-size, perform them on the client
    """
    SambaNova Inference Adapter for Llama Stack.
    """

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The SambaNova base URL
        """
        return self.config.url
