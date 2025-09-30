# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.inference import (
    Inference,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.models import Model
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import RunpodImplConfig

MODEL_ENTRIES = []


class RunpodInferenceAdapter(
    OpenAIMixin,
    ModelRegistryHelper,
    Inference,
):
    """
    Adapter for RunPod's OpenAI-compatible API endpoints.
    Supports VLLM for serverless endpoint self-hosted or public endpoints.
    Can work with any runpod endpoints that support OpenAI-compatible API
    """

    def __init__(self, config: RunpodImplConfig) -> None:
        OpenAIMixin.__init__(self)
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)
        self.config = config

    def get_api_key(self) -> str:
        """Get API key for OpenAI client."""
        return self.config.api_token

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return self.config.url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

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
    ):
        """Override to add RunPod-specific stream_options requirement."""
        if stream and not stream_options:
            stream_options = {"include_usage": True}

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

    async def register_model(self, model: Model) -> Model:
        """
        Register a model and verify it's available on the RunPod endpoint.
        In the .yaml file the model: can be defined as example
        models:
            - metadata: {}
            model_id: qwen3-32b-awq
            model_type: llm
            provider_id: runpod
            provider_model_id: Qwen/Qwen3-32B-AWQ
        """
        provider_model_id = model.provider_resource_id or model.identifier
        is_available = await self.check_model_availability(provider_model_id)

        if not is_available:
            raise ValueError(
                f"Model {provider_model_id} is not available on RunPod endpoint. "
                f"Check your RunPod endpoint configuration."
            )

        return model

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(model)
        provider_model_id = model_obj.provider_resource_id or model

        response = await self.client.embeddings.create(
            model=provider_model_id,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )

        return response
