# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Google Interactions API implementation.

Translates Google Interactions format to/from OpenAI Chat Completions format,
delegating to the inference API for actual model calls.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from llama_stack.log import get_logger
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
    GoogleStreamEvent,
    GoogleTextOutput,
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
        openai_params = self._google_to_openai(request)

        result = await self.inference_api.openai_chat_completion(openai_params)

        if isinstance(result, AsyncIterator):
            return self._stream_openai_to_google(result, request.model)

        return self._openai_to_google(result, request.model)

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
        outputs: list[GoogleTextOutput] = []

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
