# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import asyncio
import importlib
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any

from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack_api import (
    Inference,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankResponse,
)
from llama_stack_api.inference import RerankRequest

from . import converters
from .config import VertexAINativeConfig

logger = get_logger(name=__name__, category="inference::vertexai_native")


def _new_request_id() -> str:
    """Generate a unique chat-completion-style request identifier."""
    return f"chatcmpl-{uuid.uuid4()}"


class VertexAINativeInferenceAdapter(ModelRegistryHelper, NeedsRequestProviderData, Inference):
    config: VertexAINativeConfig

    def __init__(self, config: VertexAINativeConfig) -> None:
        """Initialize the adapter with provider config and an empty client cache."""
        self.config = config
        self._default_client: Any | None = None
        self._client_lock = asyncio.Lock()
        ModelRegistryHelper.__init__(self, model_entries=None, allowed_models=config.allowed_models)

    async def _get_client(self) -> Any:
        """Return a google-genai Client, respecting per-request provider data overrides."""
        project = self.config.project
        location = self.config.location

        provider_data = self.get_request_provider_data()
        if provider_data is not None:
            project = getattr(provider_data, "vertex_project", None) or project
            location = getattr(provider_data, "vertex_location", None) or location

        is_default = project == self.config.project and location == self.config.location

        if is_default and self._default_client is not None:
            return self._default_client

        genai = importlib.import_module("google.genai")

        if is_default:
            async with self._client_lock:
                # Double-check after acquiring the lock
                if self._default_client is not None:
                    return self._default_client
                self._default_client = genai.Client(vertexai=True, project=project, location=location)
                return self._default_client

        return genai.Client(vertexai=True, project=project, location=location)

    async def initialize(self) -> None:
        """No-op; the client is created lazily on first request."""
        pass

    async def shutdown(self) -> None:
        """Release the cached default client."""
        self._default_client = None

    async def list_provider_model_ids(self) -> Iterable[str]:
        """Return the static set of supported model identifiers.

        Includes native Google models and Vertex AI Model-as-a-Service (MaaS)
        partner models. MaaS models use the same google-genai SDK interface —
        the SDK handles publisher routing automatically.
        """
        return [
            # Google Gemini
            "google/gemini-2.0-flash",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            # Anthropic (MaaS)
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
            # Meta Llama (MaaS)
            "meta/llama-4-maverick-17b-128e",
            "meta/llama-4-scout-8b",
            # Mistral (MaaS)
            "mistral/mistral-medium-3",
            "mistral/mistral-small-3-1",
        ]

    async def check_model_availability(self, model: str) -> bool:
        """Return True only if the model is in the static supported list."""
        return model in await self.list_provider_model_ids()

    async def should_refresh_models(self) -> bool:
        """Always False — the model list is static and does not support refresh."""
        return False

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Not supported — Gemini models are chat-only."""
        raise NotImplementedError("Text completion not supported. Use openai_chat_completion instead.")

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Not supported — use a dedicated embeddings provider."""
        raise NotImplementedError("Embeddings not supported by vertexai_native provider.")

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Not supported — use a dedicated reranking provider."""
        raise NotImplementedError("Rerank not supported by vertexai_native provider.")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Convert an OpenAI-format chat request to google-genai and return the result."""
        provider_model_id = self.get_provider_model_id(params.model) or params.model
        request_id = _new_request_id()

        system_instruction, contents = converters.convert_messages(params.messages)
        tools, tool_config = converters.convert_tools(params.tools, params.tool_choice)
        config = converters.build_generate_config(params, system_instruction, tools, tool_config)

        ignored = converters.collect_ignored_params(params)
        if ignored:
            logger.warning(f"Ignoring unsupported params: {ignored}")

        if params.stream:
            try:
                stream = await (await self._get_client()).aio.models.generate_content_stream(
                    model=provider_model_id,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                raise RuntimeError(f"Vertex AI native chat completion failed for model '{provider_model_id}'") from e

            async def _stream() -> AsyncIterator[OpenAIChatCompletionChunk]:
                """Yield OpenAI-format chunks from the google-genai streaming response."""
                index = 0
                try:
                    async for chunk in stream:
                        yield converters.convert_stream_chunk(chunk, params.model, request_id, index)
                        index += 1
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise RuntimeError(
                        f"Vertex AI native chat completion failed for model '{provider_model_id}'"
                    ) from e
                finally:
                    if hasattr(stream, "aclose"):
                        await stream.aclose()

            return _stream()

        try:
            response = await (await self._get_client()).aio.models.generate_content(
                model=provider_model_id,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Vertex AI native chat completion failed for model '{provider_model_id}'") from e

        return converters.convert_response(response, params.model, request_id)
