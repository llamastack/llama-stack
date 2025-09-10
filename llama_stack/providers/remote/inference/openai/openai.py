# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import OpenAIConfig
from .models import MODEL_ENTRIES

logger = get_logger(name=__name__, category="inference::openai")


#
# This OpenAI adapter implements Inference methods using two mixins -
#
# | Inference Method           | Implementation Source    |
# |----------------------------|--------------------------|
# | completion                 | LiteLLMOpenAIMixin       |
# | chat_completion            | LiteLLMOpenAIMixin       |
# | embedding                  | LiteLLMOpenAIMixin       |
# | batch_completion           | LiteLLMOpenAIMixin       |
# | batch_chat_completion      | LiteLLMOpenAIMixin       |
# | openai_completion          | OpenAIMixin              |
# | openai_chat_completion     | OpenAIMixin              |
# | openai_embeddings          | OpenAIMixin              |
#
class OpenAIInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin):
    """
    OpenAI Inference Adapter for Llama Stack.

    Note: The inheritance order is important here. OpenAIMixin must come before
    LiteLLMOpenAIMixin to ensure that OpenAIMixin.check_model_availability()
    is used instead of ModelRegistryHelper.check_model_availability().

    - OpenAIMixin.check_model_availability() queries the OpenAI API to check if a model exists
    - ModelRegistryHelper.check_model_availability() (inherited by LiteLLMOpenAIMixin) just returns False and shows a warning
    """

    def __init__(self, config: OpenAIConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            litellm_provider_name="openai",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="openai_api_key",
        )
        self.config = config
        # we set is_openai_compat so users can use the canonical
        # openai model names like "gpt-4" or "gpt-3.5-turbo"
        # and the model name will be translated to litellm's
        # "openai/gpt-4" or "openai/gpt-3.5-turbo" transparently.
        # if we do not set this, users will be exposed to the
        # litellm specific model names, an abstraction leak.
        self.is_openai_compat = True

    # Delegate the client data handling get_api_key method to LiteLLMOpenAIMixin
    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        """
        Get the OpenAI API base URL.

        Returns the OpenAI API base URL from the configuration.
        """
        return self.config.base_url

    async def initialize(self) -> None:
        await super().initialize()

    async def shutdown(self) -> None:
        await super().shutdown()
