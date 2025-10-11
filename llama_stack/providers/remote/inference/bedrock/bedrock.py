# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import AsyncIterator

from botocore.client import BaseClient

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    Inference,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
)
from llama_stack.providers.remote.inference.bedrock.config import BedrockConfig
from llama_stack.providers.utils.bedrock.client import create_bedrock_client
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_strategy_options,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
)

from .models import MODEL_ENTRIES

REGION_PREFIX_MAP = {
    "us": "us.",
    "eu": "eu.",
    "ap": "ap.",
}


def _get_region_prefix(region: str | None) -> str:
    # AWS requires region prefixes for inference profiles
    if region is None:
        return "us."  # default to US when we don't know

    # Handle case insensitive region matching
    region_lower = region.lower()
    for prefix in REGION_PREFIX_MAP:
        if region_lower.startswith(f"{prefix}-"):
            return REGION_PREFIX_MAP[prefix]

    # Fallback to US for anything we don't recognize
    return "us."


def _to_inference_profile_id(model_id: str, region: str = None) -> str:
    # Return ARNs unchanged
    if model_id.startswith("arn:"):
        return model_id

    # Return inference profile IDs that already have regional prefixes
    if any(model_id.startswith(p) for p in REGION_PREFIX_MAP.values()):
        return model_id

    # Default to US East when no region is provided
    if region is None:
        region = "us-east-1"

    return _get_region_prefix(region) + model_id


class BedrockInferenceAdapter(
    ModelRegistryHelper,
    Inference,
):
    def __init__(self, config: BedrockConfig) -> None:
        ModelRegistryHelper.__init__(self, model_entries=MODEL_ENTRIES)
        self._config = config
        self._client = None

    @property
    def client(self) -> BaseClient:
        if self._client is None:
            self._client = create_bedrock_client(self._config)
        return self._client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()

    async def _get_params_for_chat_completion(self, request: ChatCompletionRequest) -> dict:
        bedrock_model = request.model

        sampling_params = request.sampling_params
        options = get_sampling_strategy_options(sampling_params)

        if sampling_params.max_tokens:
            options["max_gen_len"] = sampling_params.max_tokens
        if sampling_params.repetition_penalty > 0:
            options["repetition_penalty"] = sampling_params.repetition_penalty

        prompt = await chat_completion_request_to_prompt(request, self.get_llama_model(request.model))

        # Convert foundation model ID to inference profile ID
        region_name = self.client.meta.region_name
        inference_profile_id = _to_inference_profile_id(bedrock_model, region_name)

        return {
            "modelId": inference_profile_id,
            "body": json.dumps(
                {
                    "prompt": prompt,
                    **options,
                }
            ),
        }

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion not supported by the Bedrock provider")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError("OpenAI chat completion not supported by the Bedrock provider")
