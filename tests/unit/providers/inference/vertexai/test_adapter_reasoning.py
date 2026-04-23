# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for VertexAI openai_chat_completions_with_reasoning()."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from llama_stack_api.inference.models import (
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
)

from .conftest import _make_fake_streaming_chunk


def _make_thinking_streaming_chunk(
    text: str = "answer",
    thinking_text: str = "let me think",
) -> SimpleNamespace:
    """Build a fake streaming chunk with both a thinking part and a text part."""
    thinking_part = SimpleNamespace(text=thinking_text, thought=True, function_call=None)
    text_part = SimpleNamespace(text=text, thought=None, function_call=None)
    content = SimpleNamespace(parts=[thinking_part, text_part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP", index=0, logprobs_result=None)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


@pytest.fixture
def streaming_adapter(adapter, monkeypatch):
    """Adapter wired to stream from a list of fake Gemini chunks.

    Returns a callable: ``streaming_adapter(chunks)`` patches the adapter
    and returns it ready for ``openai_chat_completions_with_reasoning()``.
    """

    def _configure(chunks: list[SimpleNamespace]):
        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            for c in chunks:
                yield c

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )

        async def _provider_model_id(_: str) -> str:
            """Return a fixed provider model identifier."""
            return "gemini-2.5-flash"

        monkeypatch.setattr(adapter, "_get_provider_model_id", _provider_model_id)
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        return adapter

    return _configure


def _make_streaming_params(**overrides: Any) -> OpenAIChatCompletionRequestWithExtraBody:
    """Build a minimal streaming chat completion request."""
    defaults = {
        "model": "google/gemini-2.5-flash",
        "messages": cast(Any, [{"role": "user", "content": "hi"}]),
        "stream": True,
    }
    defaults.update(overrides)
    return OpenAIChatCompletionRequestWithExtraBody(**defaults)


class TestOpenAIChatCompletionsWithReasoning:
    """Test the openai_chat_completions_with_reasoning method."""

    async def test_non_streaming_raises_not_implemented(self, adapter, patch_chat_completion_dependencies):
        """Non-streaming reasoning raises NotImplementedError."""
        patch_chat_completion_dependencies(adapter)
        params = _make_streaming_params(stream=False, reasoning_effort="medium")
        with pytest.raises(NotImplementedError, match="Non-streaming reasoning"):
            await adapter.openai_chat_completions_with_reasoning(params)

    async def test_all_chunks_wrapped(self, streaming_adapter):
        """Every yielded chunk is an OpenAIChatCompletionChunkWithReasoning."""
        adapter = streaming_adapter(
            [
                _make_thinking_streaming_chunk(text="Hello", thinking_text="Planning"),
                _make_fake_streaming_chunk("world"),
            ]
        )

        result = await adapter.openai_chat_completions_with_reasoning(
            _make_streaming_params(reasoning_effort="high"),
        )
        chunks = [c async for c in result]

        assert len(chunks) == 2
        assert all(isinstance(c, OpenAIChatCompletionChunkWithReasoning) for c in chunks)

    @pytest.mark.parametrize(
        "gemini_chunks, expected_reasoning",
        [
            pytest.param(
                [_make_thinking_streaming_chunk(text="Hi", thinking_text="Planning response")],
                ["Planning response"],
                id="thinking_part_extracted",
            ),
            pytest.param(
                [_make_fake_streaming_chunk("just text")],
                [None],
                id="no_thinking_yields_none",
            ),
            pytest.param(
                [
                    _make_thinking_streaming_chunk(text="a", thinking_text="thought1"),
                    _make_fake_streaming_chunk("b"),
                    _make_thinking_streaming_chunk(text="c", thinking_text="thought2"),
                ],
                ["thought1", None, "thought2"],
                id="mixed_thinking_and_plain",
            ),
        ],
    )
    async def test_reasoning_content_extraction(self, streaming_adapter, gemini_chunks, expected_reasoning):
        """Reasoning content is correctly extracted (or None) per chunk."""
        adapter = streaming_adapter(gemini_chunks)

        result = await adapter.openai_chat_completions_with_reasoning(_make_streaming_params())
        chunks = [c async for c in result]

        actual = [c.reasoning_content for c in chunks]
        assert actual == expected_reasoning

    async def test_inner_chunk_preserves_model_and_role(self, streaming_adapter):
        """The .chunk attribute contains the original OpenAIChatCompletionChunk unchanged."""
        adapter = streaming_adapter([_make_thinking_streaming_chunk(text="answer", thinking_text="thinking")])

        result = await adapter.openai_chat_completions_with_reasoning(_make_streaming_params())
        wrapped = [c async for c in result]

        inner = wrapped[0].chunk
        assert inner.model == "google/gemini-2.5-flash"
        assert inner.choices[0].delta.role == "assistant"
        assert inner.choices[0].delta.content == "answer"
