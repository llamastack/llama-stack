# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable
from typing import Any

from openai import AuthenticationError

from llama_stack.apis.inference import (
    Model,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.telemetry.tracing import get_current_span

from .config import BedrockConfig


class BedrockInferenceAdapter(OpenAIMixin):
    """
    Adapter for AWS Bedrock's OpenAI-compatible API endpoints.

    Supports Llama models across regions and GPT-OSS models (us-west-2 only).

    Note: Bedrock's OpenAI-compatible endpoint does not support /v1/models
    for dynamic model discovery. Models must be pre-registered in the config.
    """

    config: BedrockConfig
    provider_data_api_key_field: str = "aws_bedrock_api_key"

    def get_api_key(self) -> str:
        """Get API key for OpenAI client."""
        if not self.config.api_key:
            raise ValueError(
                "API key is not set. Please provide a valid API key in the "
                "provider config or via AWS_BEDROCK_API_KEY environment variable."
            )
        return self.config.api_key

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return f"https://bedrock-runtime.{self.config.region_name}.amazonaws.com/openai/v1"

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        Bedrock's OpenAI-compatible endpoint does not support the /v1/models endpoint.
        Returns empty list since models must be pre-registered in the config.
        """
        return []

    async def register_model(self, model: Model) -> Model:
        """
        Register a model with the Bedrock provider.

        Bedrock doesn't support dynamic model listing via /v1/models, so we skip
        the availability check and accept all models registered in the config.
        """
        return model

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        """Bedrock's OpenAI-compatible API does not support the /v1/embeddings endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/embeddings endpoint. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_completion(
        self,
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        suffix: str | None = None,
    ) -> OpenAICompletion:
        """Bedrock's OpenAI-compatible API does not support the /v1/completions endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/completions endpoint. "
            "Only /v1/chat/completions is supported. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_chat_completion(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        frequency_penalty: float | None = None,
        function_call: str | dict[str, Any] | None = None,
        functions: list[dict[str, Any]] | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: OpenAIResponseFormatParam | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Override to enable streaming usage metrics and handle authentication errors."""
        # Enable streaming usage metrics when telemetry is active
        if stream and get_current_span() is not None:
            if stream_options is None:
                stream_options = {"include_usage": True}
            elif "include_usage" not in stream_options:
                stream_options = {**stream_options, "include_usage": True}

        # Wrap call in try/except to catch authentication errors
        try:
            return await super().openai_chat_completion(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                function_call=function_call,
                functions=functions,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream=stream,
                stream_options=stream_options,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
            )
        except AuthenticationError as e:
            raise ValueError(
                f"AWS Bedrock authentication failed: {e.message}. "
                "Please check your API key in the provider config or x-llamastack-provider-data header."
            ) from e
