# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

import httpx
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from pydantic import ConfigDict

from llama_stack.apis.inference import (
    OpenAIChatCompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    ToolChoice,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import VLLMInferenceAdapterConfig

log = get_logger(name=__name__, category="inference::vllm")


class VLLMInferenceAdapter(OpenAIMixin):
    config: VLLMInferenceAdapterConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_data_api_key_field: str = "vllm_api_token"

    def get_api_key(self) -> str:
        return self.config.api_token or ""

    def get_base_url(self) -> str:
        """Get the base URL from config."""
        if not self.config.url:
            raise ValueError("No base URL configured")
        return self.config.url

    async def initialize(self) -> None:
        if not self.config.url:
            raise ValueError(
                "You must provide a URL in run.yaml (or via the VLLM_URL environment variable) to use vLLM."
            )

    async def should_refresh_models(self) -> bool:
        # Strictly respecting the refresh_models directive
        return self.config.refresh_models

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the remote vLLM server.
        This method is used by the Provider API to verify
        that the service is running correctly.
        Uses the unauthenticated /health endpoint.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            base_url = self.get_base_url()
            health_url = urljoin(base_url, "health")

            async with httpx.AsyncClient() as client:
                response = await client.get(health_url)
                response.raise_for_status()
                return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    def get_extra_client_params(self):
        return {"http_client": httpx.AsyncClient(verify=self.config.tls_verify)}

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
        max_tokens = max_tokens or self.config.max_tokens

        # This is to be consistent with OpenAI API and support vLLM <= v0.6.3
        # References:
        #   * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        #   * https://github.com/vllm-project/vllm/pull/10000
        if not tools and tool_choice is not None:
            tool_choice = ToolChoice.none.value

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
