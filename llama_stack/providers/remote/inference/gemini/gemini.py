# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import GeminiConfig


class GeminiInferenceAdapter(OpenAIMixin):
    config: GeminiConfig

    provider_data_api_key_field: str = "gemini_api_key"
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-004": {"embedding_dimension": 768, "context_length": 2048},
    }

    def get_api_key(self) -> str:
        return self.config.api_key or ""

    def get_base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
