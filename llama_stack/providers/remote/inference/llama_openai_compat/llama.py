# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .models import MODEL_ENTRIES

logger = get_logger(name=__name__, category="inference::llama_openai_compat")


class LlamaCompatInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    """
    Llama API Inference Adapter for Llama Stack.

    Note: The inheritance order is important here. OpenAIMixin must come before
    LiteLLMOpenAIMixin to ensure that OpenAIMixin.check_model_availability()
    is used instead of ModelRegistryHelper.check_model_availability().

    - OpenAIMixin.check_model_availability() queries the Llama API to check if a model exists
    - ModelRegistryHelper.check_model_availability() (inherited by LiteLLMOpenAIMixin) just returns False and shows a warning
    """

    _config: LlamaCompatConfig

    def __init__(self, config: LlamaCompatConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            litellm_provider_name="meta_llama",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="llama_api_key",
            openai_compat_api_base=config.openai_compat_api_base,
        )
        self.config = config

    # Delegate the client data handling get_api_key method to LiteLLMOpenAIMixin
    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The Llama API base URL
        """
        return self.config.openai_compat_api_base

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()
