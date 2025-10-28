# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import OpenAIConfig

logger = get_logger(name=__name__, category="inference::openai")


#
# This OpenAI adapter implements Inference methods using OpenAIMixin
#
class OpenAIInferenceAdapter(OpenAIMixin):
    """
    OpenAI Inference Adapter for Llama Stack.
    """

    config: OpenAIConfig

    provider_data_api_key_field: str = "openai_api_key"

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-3-large": {"embedding_dimension": 3072, "context_length": 8192},
    }

    def get_base_url(self) -> str:
        """
        Get the OpenAI API base URL.

        Returns the OpenAI API base URL from the configuration.
        """
        return self.config.base_url
