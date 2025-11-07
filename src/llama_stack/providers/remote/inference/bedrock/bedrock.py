# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable

from openai import AuthenticationError

from llama_stack.apis.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.core.telemetry.tracing import get_current_span
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import BedrockConfig

logger = get_logger(name=__name__, category="inference::bedrock")


class BedrockInferenceAdapter(OpenAIMixin):
    """
    Adapter for AWS Bedrock's OpenAI-compatible API endpoints.

    Supports Llama models across regions and GPT-OSS models (us-west-2 only).

    Note: Bedrock's OpenAI-compatible endpoint does not support /v1/models
    for dynamic model discovery. Models must be pre-registered in the config.
    """

    config: BedrockConfig
    provider_data_api_key_field: str = "aws_bedrock_api_key"

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return f"https://bedrock-runtime.{self.config.region_name}.amazonaws.com/openai/v1"

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        Bedrock's OpenAI-compatible endpoint does not support the /v1/models endpoint.
        Returns empty list since models must be pre-registered in the config.
        """
        return []

    async def check_model_availability(self, model: str) -> bool:
        """
        Bedrock doesn't support dynamic model listing via /v1/models.
        Always return True to accept all models registered in the config.
        """
        return True

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Bedrock's OpenAI-compatible API does not support the /v1/embeddings endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/embeddings endpoint. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        """Bedrock's OpenAI-compatible API does not support the /v1/completions endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/completions endpoint. "
            "Only /v1/chat/completions is supported. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Override to enable streaming usage metrics and handle authentication errors."""
        # Enable streaming usage metrics when telemetry is active
        if params.stream and get_current_span() is not None:
            if params.stream_options is None:
                params.stream_options = {"include_usage": True}
            elif "include_usage" not in params.stream_options:
                params.stream_options = {**params.stream_options, "include_usage": True}

        try:
            logger.debug(f"Calling Bedrock OpenAI API with model={params.model}, stream={params.stream}")
            result = await super().openai_chat_completion(params=params)
            logger.debug(f"Bedrock API returned: {type(result).__name__ if result is not None else 'None'}")

            if result is None:
                logger.error(f"Bedrock OpenAI client returned None for model={params.model}, stream={params.stream}")
                raise RuntimeError(
                    f"Bedrock API returned no response for model '{params.model}'. "
                    "This may indicate the model is not supported or a network/API issue occurred."
                )

            return result
        except AuthenticationError as e:
            error_msg = str(e)

            # Check if this is a token expiration error
            if "expired" in error_msg.lower() or "Bearer Token has expired" in error_msg:
                logger.error(f"AWS Bedrock authentication token expired: {error_msg}")
                raise ValueError(
                    "AWS Bedrock authentication failed: Bearer token has expired. "
                    "The AWS_BEDROCK_API_KEY environment variable contains an expired pre-signed URL. "
                    "Please refresh your token by generating a new pre-signed URL with AWS credentials. "
                    "Refer to AWS Bedrock documentation for details on OpenAI-compatible endpoints."
                ) from e
            else:
                logger.error(f"AWS Bedrock authentication failed: {error_msg}")
                raise ValueError(
                    f"AWS Bedrock authentication failed: {error_msg}. "
                    "Please verify your API key is correct in the provider config or x-llamastack-provider-data header. "
                    "The API key should be a valid AWS pre-signed URL for Bedrock's OpenAI-compatible endpoint."
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock API: {type(e).__name__}: {e}", exc_info=True)
            raise
