# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from urllib.parse import urljoin

from llama_stack.apis.inference import OpenAIEmbeddingsResponse
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import CerebrasImplConfig


class CerebrasInferenceAdapter(OpenAIMixin):
    config: CerebrasImplConfig

    def get_api_key(self) -> str:
        return self.config.api_key.get_secret_value()

    def get_base_url(self) -> str:
        return urljoin(self.config.base_url, "v1")

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
