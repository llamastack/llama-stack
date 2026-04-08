# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
from collections.abc import AsyncIterator
from typing import cast

from ollama import AsyncClient as AsyncOllamaClient

from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.ollama.config import OllamaImplConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.reasoning import map_reasoning_messages, wrap_chunks_with_reasoning
from llama_stack_api import (
    HealthResponse,
    HealthStatus,
    Model,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
    UnsupportedModelError,
)

logger = get_logger(name=__name__, category="inference::ollama")


class OllamaInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Ollama local model runtime."""

    config: OllamaImplConfig

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "all-minilm:l6-v2": {
            "embedding_dimension": 384,
            "context_length": 512,
        },
        "nomic-embed-text:latest": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
        "nomic-embed-text:v1.5": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
        "nomic-embed-text:137m-v1.5-fp16": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
    }

    download_images: bool = True
    _clients: dict[asyncio.AbstractEventLoop, AsyncOllamaClient] = {}

    @property
    def ollama_client(self) -> AsyncOllamaClient:
        # ollama client attaches itself to the current event loop (sadly?)
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            # Ollama client expects base URL without /v1 suffix
            base_url_str = str(self.config.base_url)
            if base_url_str.endswith("/v1"):
                host = base_url_str[:-3]
            else:
                host = base_url_str
            self._clients[loop] = AsyncOllamaClient(host=host)
        return self._clients[loop]

    def get_api_key(self):
        return "NO KEY REQUIRED"

    def get_base_url(self):
        return str(self.config.base_url)

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Chat completion with reasoning support for Ollama.

        Extracts reasoning from Ollama's response and wraps it in internal
        OpenAIChatCompletionChunkWithReasoning so the Responses layer can
        read reasoning as a typed field.
        """
        if not params.stream:
            raise NotImplementedError("Non-streaming reasoning is not yet supported for Ollama")

        params = params.model_copy()
        if params.reasoning_effort is None:
            params.reasoning_effort = "none"
        params.messages = map_reasoning_messages(params.messages, reasoning_field="reasoning")
        result = await self.openai_chat_completion(params)
        result = cast(AsyncIterator[OpenAIChatCompletionChunk], result)
        return wrap_chunks_with_reasoning(result, reasoning_fields=("reasoning", "reasoning_content"))

    async def initialize(self) -> None:
        logger.info("checking connectivity to Ollama", base_url=self.config.base_url)
        r = await self.health()
        if r["status"] == HealthStatus.ERROR:
            logger.warning(
                "Ollama Server is not running (message: ). Make sure to start it using `ollama serve` in a separate terminal",
                r_message=r["message"],
            )

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the Ollama server.
        This method is used by initialize() and the Provider API to verify that the service is running
        correctly.
        Returns:
            HealthResponse: A dictionary containing the health status.
        """
        try:
            await self.ollama_client.ps()
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def shutdown(self) -> None:
        self._clients.clear()

    async def register_model(self, model: Model) -> Model:
        if await self.check_model_availability(model.provider_model_id):
            return model
        elif await self.check_model_availability(f"{model.provider_model_id}:latest"):
            model.provider_resource_id = f"{model.provider_model_id}:latest"
            logger.warning(
                "Imprecise provider resource id was used but 'latest' is available in Ollama - using",
                provider_model_id=model.provider_model_id,
            )
            return model

        raise UnsupportedModelError(model.provider_model_id, list(self._model_cache.keys()))
