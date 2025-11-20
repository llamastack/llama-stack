# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack_api import (
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

from .config import CerebrasImplConfig


class CerebrasInferenceAdapter(OpenAIMixin):
    config: CerebrasImplConfig

    provider_data_api_key_field: str = "cerebras_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
