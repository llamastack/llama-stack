# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import SambaNovaImplConfig
from .models import MODEL_ENTRIES


class SambaNovaInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    """
    SambaNova Inference Adapter for Llama Stack.

    Note: The inheritance order is important here. OpenAIMixin must come before
    LiteLLMOpenAIMixin to ensure that OpenAIMixin.check_model_availability()
    is used instead of LiteLLMOpenAIMixin.check_model_availability().

    - OpenAIMixin.check_model_availability() queries the /v1/models to check if a model exists
    - LiteLLMOpenAIMixin.check_model_availability() checks the static registry within LiteLLM
    """

    def __init__(self, config: SambaNovaImplConfig):
        self.config = config
        self.environment_available_models = []
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            litellm_provider_name="sambanova",
            api_key_from_config=self.config.api_key.get_secret_value() if self.config.api_key else None,
            provider_data_api_key_field="sambanova_api_key",
            openai_compat_api_base=self.config.url,
            download_images=True,  # SambaNova requires base64 image encoding
            json_schema_strict=False,  # SambaNova doesn't support strict=True yet
        )

    # Delegate the client data handling get_api_key method to LiteLLMOpenAIMixin
    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The SambaNova base URL
        """
        return self.config.url
