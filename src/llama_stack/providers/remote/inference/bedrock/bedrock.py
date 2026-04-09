# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from openai import AuthenticationError

from llama_stack.log import get_logger
from llama_stack.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionWithReasoning,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

from .config import BedrockConfig

logger = get_logger(name=__name__, category="inference::bedrock")


class BedrockInferenceAdapter(OpenAIMixin):
    """
    Adapter for AWS Bedrock's OpenAI-compatible API endpoints.

    Supports Llama models across regions and GPT-OSS models (us-west-2 only).
    """

    config: BedrockConfig
    provider_data_api_key_field: str = "aws_bearer_token_bedrock"

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return f"https://bedrock-mantle.{self.config.region_name}.api.aws/v1"

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Bedrock's OpenAI-compatible API does not support the /v1/embeddings endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/embeddings endpoint. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    def _prepare_reasoning_params(self, params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        """Adapt CC request params to match what Bedrock expects for reasoning.

        No-op for now. Override if Bedrock needs specific param adjustments.
        """
        pass

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Chat completion with reasoning support for Bedrock.

        Extracts reasoning from Bedrock's response and wraps it in internal
        types so the Responses layer can read reasoning as a typed field.
        """
        if not params.stream:
            raise NotImplementedError("Non-streaming reasoning is not yet supported for Bedrock")

        params = params.model_copy()
        self._prepare_reasoning_params(params)

        # Bedrock's CC endpoint expects 'reasoning' on assistant messages, but
        # that field isn't part of the official CC spec. Convert to dicts so we
        # can rename reasoning_content → reasoning.
        mapped_messages: list = []
        for msg in params.messages:
            if isinstance(msg, AssistantMessageWithReasoning) and msg.reasoning_content:
                msg_dict = msg.model_dump(exclude_none=True)
                msg_dict["reasoning"] = msg_dict.pop("reasoning_content")
                mapped_messages.append(msg_dict)
            else:
                mapped_messages.append(msg)
        params.messages = mapped_messages

        result = await self.openai_chat_completion(params)

        async def _wrap_chunks() -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
            async for chunk in result:
                reasoning = None
                for choice in chunk.choices or []:
                    reasoning = getattr(choice.delta, "reasoning", None) or getattr(
                        choice.delta, "reasoning_content", None
                    )
                yield OpenAIChatCompletionChunkWithReasoning(
                    chunk=chunk,
                    reasoning_content=reasoning,
                )

        return _wrap_chunks()

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Override to handle authentication errors and null responses."""
        try:
            logger.debug("Calling Bedrock OpenAI API", model=params.model, stream=params.stream)
            result = await super().openai_chat_completion(params=params)
            logger.debug("Bedrock API returned", result_type=type(result).__name__ if result is not None else "None")

            if result is None:
                logger.error("Bedrock OpenAI client returned None", model=params.model, stream=params.stream)
                raise RuntimeError(
                    f"Bedrock API returned no response for model '{params.model}'. "
                    "This may indicate the model is not supported or a network/API issue occurred."
                )

            return result
        except AuthenticationError as e:
            error_msg = str(e)

            # Check if this is a token expiration error
            if "expired" in error_msg.lower() or "Bearer Token has expired" in error_msg:
                logger.error("AWS Bedrock authentication token expired", error=error_msg)
                raise ValueError(
                    "AWS Bedrock authentication failed: Bearer token has expired. "
                    "The AWS_BEARER_TOKEN_BEDROCK environment variable contains an expired pre-signed URL. "
                    "Please refresh your token by generating a new pre-signed URL with AWS credentials. "
                    "Refer to AWS Bedrock documentation for details on OpenAI-compatible endpoints."
                ) from e
            else:
                logger.error("AWS Bedrock authentication failed", error=error_msg)
                raise ValueError(
                    f"AWS Bedrock authentication failed: {error_msg}. "
                    "Please verify your API key is correct in the provider config or x-llamastack-provider-data header. "
                    "The API key should be a valid AWS pre-signed URL for Bedrock's OpenAI-compatible endpoint."
                ) from e
        except Exception as e:
            logger.error(
                "Unexpected error calling Bedrock API", error_type=type(e).__name__, error=str(e), exc_info=True
            )
            raise
