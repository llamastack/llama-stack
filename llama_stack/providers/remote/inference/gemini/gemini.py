# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from openai import NOT_GIVEN

from llama_stack.apis.inference import (
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import GeminiConfig


class GeminiInferenceAdapter(OpenAIMixin):
    config: GeminiConfig

    provider_data_api_key_field: str = "gemini_api_key"
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "models/text-embedding-004": {"embedding_dimension": 768, "context_length": 2048},
        "models/gemini-embedding-001": {"embedding_dimension": 3072, "context_length": 2048},
    }

    def get_base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta/openai/"

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """
        Override embeddings method to handle Gemini's missing usage statistics.
        Gemini's embedding API doesn't return usage information, so we provide default values.
        """
        # Prepare request parameters
        request_params = {
            "model": await self._get_provider_model_id(params.model),
            "input": params.input,
            "encoding_format": params.encoding_format if params.encoding_format is not None else NOT_GIVEN,
            "dimensions": params.dimensions if params.dimensions is not None else NOT_GIVEN,
            "user": params.user if params.user is not None else NOT_GIVEN,
        }

        # Add extra_body if present
        extra_body = params.model_extra
        if extra_body:
            request_params["extra_body"] = extra_body

        # Call OpenAI embeddings API with properly typed parameters
        response = await self.client.embeddings.create(**request_params)

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        # Gemini doesn't return usage statistics - use default values
        if hasattr(response, "usage") and response.usage:
            usage = OpenAIEmbeddingUsage(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            usage = OpenAIEmbeddingUsage(
                prompt_tokens=0,
                total_tokens=0,
            )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=params.model,
            usage=usage,
        )
