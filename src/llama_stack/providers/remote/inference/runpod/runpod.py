# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from llama_stack.apis.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import RunpodImplConfig


class RunpodInferenceAdapter(OpenAIMixin):
    """
    Adapter for RunPod's OpenAI-compatible API endpoints.
    Supports VLLM for serverless endpoint self-hosted or public endpoints.
    Can work with any runpod endpoints that support OpenAI-compatible API
    """

    config: RunpodImplConfig
    provider_data_api_key_field: str = "runpod_api_token"

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return self.config.url

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Override to add RunPod-specific stream_options requirement."""
        params = params.model_copy()

        if params.stream and not params.stream_options:
            params.stream_options = {"include_usage": True}

        return await super().openai_chat_completion(params)
