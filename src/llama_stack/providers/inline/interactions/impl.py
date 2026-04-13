# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Google Interactions API implementation.

Translates Google Interactions format to/from OpenAI Chat Completions format,
delegating to the inference API for actual model calls. When the underlying
inference provider natively supports the Google Interactions API (e.g. Gemini),
requests are forwarded directly without translation.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, TypedDict

import httpx

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.http_client import _build_network_client_kwargs
from llama_stack.providers.utils.inference.model_registry import NetworkConfig
from llama_stack_api import (
    Inference,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
)
from llama_stack_api.interactions import Interactions
from llama_stack_api.interactions.models import (
    ContentDeltaEvent,
    ContentStartEvent,
    ContentStopEvent,
    GoogleCreateInteractionRequest,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleInteractionResponse,
    GoogleOutput,
    GoogleStreamEvent,
    GoogleTextOutput,
    GoogleThoughtOutput,
    GoogleUsage,
    InteractionCompleteEvent,
    InteractionStartEvent,
    _ContentRef,
    _InteractionCompleteRef,
    _InteractionRef,
    _TextDelta,
)

from .config import InteractionsConfig

logger = get_logger(name=__name__, category="interactions")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S+00:00")


class _PassthroughInfo(TypedDict):
    base_url: str
    api_key: str
    provider_resource_id: str
    network_config: NetworkConfig | None


class BuiltinInteractionsImpl(Interactions):
    """Google Interactions API adapter that translates to the inference API."""

    def __init__(self, config: InteractionsConfig, inference_api: Inference):
        self.config = config
        self.inference_api = inference_api

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def create_interaction(
        self,
        request: GoogleCreateInteractionRequest,
    ) -> GoogleInteractionResponse | AsyncIterator[GoogleStreamEvent]:
        # Stream translation currently supports the subset of events exposed by GoogleStreamEvent.
        # Native Gemini streaming can emit additional event/content types, so only passthrough
        # non-streaming requests for now.
        passthrough = await self._get_passthrough_info(request.model) if not request.stream else None
        if passthrough:
            return await self._passthrough_request(passthrough, request)

        openai_params = self._google_to_openai(request)

        result = await self.inference_api.openai_chat_completion(openai_params)

        if isinstance(result, AsyncIterator):
            return self._stream_openai_to_google(result, request.model)

        return self._openai_to_google(result, request.model)

    # -- Native passthrough for providers with /interactions support --

    # Module paths of provider impls known to support /interactions natively
    _NATIVE_INTERACTIONS_MODULES = {"llama_stack.providers.remote.inference.gemini"}

    async def _get_passthrough_info(self, model: str) -> _PassthroughInfo | None:
        """Check if the model's provider supports /interactions natively.

        Returns a dict with 'base_url' and 'api_key' for passthrough, or None to use translation.
        """
        routing_table = getattr(self.inference_api, "routing_table", None)
        if routing_table is None:
            return None

        obj = await routing_table.get_object_by_identifier("model", model)
        if obj is None:
            return None

        provider_impl = await routing_table.get_provider_impl(obj.identifier)
        provider_module = type(provider_impl).__module__
        is_native = any(provider_module.startswith(m) for m in self._NATIVE_INTERACTIONS_MODULES)

        if is_native and hasattr(provider_impl, "get_base_url"):
            base_url = str(provider_impl.get_base_url()).rstrip("/")
            # The Gemini provider returns a URL like
            # https://generativelanguage.googleapis.com/v1beta/openai
            # Strip the /openai suffix — Interactions sits alongside it at /v1beta/interactions
            if base_url.endswith("/openai"):
                base_url = base_url[: -len("/openai")]

            api_key = None
            if hasattr(provider_impl, "_get_api_key_from_config_or_provider_data"):
                api_key = provider_impl._get_api_key_from_config_or_provider_data()

            if not api_key:
                logger.debug("No API key for passthrough, falling back to translation", model=model)
                return None

            provider_config = getattr(provider_impl, "config", None)
            network_config = getattr(provider_config, "network", None)
            logger.info("Using native /interactions passthrough", model=model, base_url=base_url)
            return {
                "base_url": base_url,
                "api_key": api_key,
                "provider_resource_id": obj.provider_resource_id,
                "network_config": network_config,
            }

        return None

    def _build_passthrough_client_kwargs(self, passthrough: _PassthroughInfo) -> dict[str, Any]:
        client_kwargs = _build_network_client_kwargs(passthrough["network_config"])
        headers = dict(client_kwargs.get("headers", {}))
        headers.update(
            {
                "content-type": "application/json",
                "x-goog-api-key": passthrough["api_key"],
            }
        )
        client_kwargs["headers"] = headers
        client_kwargs.setdefault("timeout", httpx.Timeout(300.0))
        return client_kwargs

    async def _passthrough_request(
        self,
        passthrough: _PassthroughInfo,
        request: GoogleCreateInteractionRequest,
    ) -> GoogleInteractionResponse | AsyncIterator[GoogleStreamEvent]:
        """Forward the request directly to the provider's /interactions endpoint."""
        base_url = passthrough["base_url"]
        provider_model = passthrough["provider_resource_id"]

        url = f"{base_url}/interactions"
        body = request.model_dump(exclude_none=True)
        body["model"] = provider_model

        client_kwargs = self._build_passthrough_client_kwargs(passthrough)

        if request.stream:
            return self._passthrough_stream(url, body, client_kwargs)

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return GoogleInteractionResponse(**resp.json())

    async def _passthrough_stream(
        self,
        url: str,
        body: dict[str, Any],
        client_kwargs: dict[str, Any],
    ) -> AsyncIterator[GoogleStreamEvent]:
        """Stream SSE events directly from the provider."""
        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream("POST", url, json=body) as resp:
                resp.raise_for_status()
                event_type = None
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: ") and event_type:
                        data = json.loads(line[6:])
                        event = self._parse_sse_event(event_type, data)
                        if event:
                            yield event
                        event_type = None

    def _parse_sse_event(self, event_type: str, data: dict[str, Any]) -> GoogleStreamEvent | None:
        """Parse a Google Interactions SSE event from its type and data."""
        if event_type == "interaction.start":
            return InteractionStartEvent(interaction=_InteractionRef(**data.get("interaction", data)))
        if event_type == "content.start":
            return ContentStartEvent(index=data.get("index", 0), content=_ContentRef())
        if event_type == "content.delta":
            delta_data = data.get("delta", {})
            return ContentDeltaEvent(index=data.get("index", 0), delta=_TextDelta(text=delta_data.get("text", "")))
        if event_type == "content.stop":
            return ContentStopEvent(index=data.get("index", 0))
        if event_type == "interaction.complete":
            return InteractionCompleteEvent(
                interaction=_InteractionCompleteRef(**data.get("interaction", data)),
            )
        return None

    # -- Request translation --

    def _google_to_openai(self, request: GoogleCreateInteractionRequest) -> OpenAIChatCompletionRequestWithExtraBody:
        messages = self._convert_input_to_openai(request.system_instruction, request.input)
        gen_config = request.generation_config or GoogleGenerationConfig()

        extra_body: dict[str, Any] = {}
        if gen_config.top_k is not None:
            extra_body["top_k"] = gen_config.top_k

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=request.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=gen_config.max_output_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            stream=request.stream or False,
            **(extra_body or {}),
        )
        return params

    def _convert_input_to_openai(
        self,
        system_instruction: str | None,
        input_data: str | list[GoogleInputTurn],
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []

        if system_instruction is not None:
            openai_messages.append({"role": "system", "content": system_instruction})

        if isinstance(input_data, str):
            openai_messages.append({"role": "user", "content": input_data})
        else:
            for turn in input_data:
                role = "assistant" if turn.role == "model" else turn.role
                # Extract text from content items
                text = "\n".join(item.text for item in turn.content)
                openai_messages.append({"role": role, "content": text})

        return openai_messages

    # -- Response translation --

    def _openai_to_google(self, response: OpenAIChatCompletion, request_model: str) -> GoogleInteractionResponse:
        outputs: list[GoogleTextOutput | GoogleThoughtOutput | GoogleOutput] = []

        if response.choices:
            choice = response.choices[0]
            message = choice.message

            if message and message.content:
                outputs.append(GoogleTextOutput(text=message.content))

        usage = GoogleUsage()
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            usage = GoogleUsage(
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

        now = _now_iso()
        return GoogleInteractionResponse(
            id=f"interaction-{uuid.uuid4().hex[:24]}",
            created=now,
            updated=now,
            model=request_model,
            outputs=outputs,
            usage=usage,
        )

    # -- Streaming translation --

    async def _stream_openai_to_google(
        self,
        openai_stream: AsyncIterator[OpenAIChatCompletionChunk],
        request_model: str,
    ) -> AsyncIterator[GoogleStreamEvent]:
        """Translate OpenAI streaming chunks to Google streaming events."""

        interaction_id = f"interaction-{uuid.uuid4().hex[:24]}"

        # Emit interaction.start
        yield InteractionStartEvent(
            interaction=_InteractionRef(
                id=interaction_id,
                model=request_model,
            ),
        )

        content_started = False
        output_tokens = 0
        input_tokens = 0

        async for chunk in openai_stream:
            if not chunk.choices:
                # Usage-only chunk
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                if not content_started:
                    yield ContentStartEvent(index=0, content=_ContentRef())
                    content_started = True

                yield ContentDeltaEvent(
                    index=0,
                    delta=_TextDelta(text=delta.content),
                )

            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

        # Close content block if opened
        if content_started:
            yield ContentStopEvent(index=0)

        # Final event
        now = _now_iso()
        yield InteractionCompleteEvent(
            interaction=_InteractionCompleteRef(
                id=interaction_id,
                created=now,
                updated=now,
                model=request_model,
                usage=GoogleUsage(
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            ),
        )
