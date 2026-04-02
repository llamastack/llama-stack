# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared reasoning utilities for inference providers."""

from collections.abc import AsyncIterator

from llama_stack.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
from llama_stack_api import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
)
from llama_stack_api.inference.models import OpenAIMessageParam


def map_reasoning_messages(
    messages: list[OpenAIMessageParam],
    reasoning_field: str,
) -> list:
    """Convert AssistantMessageWithReasoning to dicts with provider-expected field name."""
    mapped: list = []
    for msg in messages:
        if isinstance(msg, AssistantMessageWithReasoning) and msg.reasoning_content:
            msg_dict = msg.model_dump(exclude_none=True)
            msg_dict[reasoning_field] = msg_dict.pop("reasoning_content")
            mapped.append(msg_dict)
        else:
            mapped.append(msg)
    return mapped


async def wrap_chunks_with_reasoning(
    chunks: AsyncIterator[OpenAIChatCompletionChunk],
    reasoning_fields: tuple[str, ...],
) -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
    """Extract reasoning from streaming CC chunks and wrap in internal type."""
    async for chunk in chunks:
        reasoning = None
        for choice in chunk.choices or []:
            for field in reasoning_fields:
                reasoning = getattr(choice.delta, field, None)
                if reasoning:
                    break
        yield OpenAIChatCompletionChunkWithReasoning(chunk=chunk, reasoning_content=reasoning)
