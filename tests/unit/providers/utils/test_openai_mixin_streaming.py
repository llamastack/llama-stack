# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Regression tests for streaming behavior in OpenAIMixin._maybe_overwrite_id().

- Issue #3185: AsyncStream passed where AsyncIterator expected.
- Issue #5122: Gemini overcounts tokens when usage is returned in every chunk.
"""

import inspect
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin


class MockAsyncStream:
    """Simulates OpenAI SDK's AsyncStream: has close() but NOT aclose()."""

    def __init__(self, chunks):
        self.chunks = chunks
        self._iter = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e

    async def close(self):
        pass


class MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockChunk:
    def __init__(self, chunk_id: str, content: str = "test", usage: MockUsage | None = None):
        self.id = chunk_id
        self.content = content
        self.usage = usage


class OpenAIMixinTestImpl(OpenAIMixin):
    __provider_id__: str = "test-provider"

    def get_api_key(self) -> str:
        return "test-api-key"

    def get_base_url(self) -> str:
        return "http://test-base-url"


@pytest.fixture
def mixin():
    config = RemoteInferenceProviderConfig()
    m = OpenAIMixinTestImpl(config=config)
    m.overwrite_completion_id = False
    return m


class TestIssue3185Regression:
    async def test_streaming_result_has_aclose(self, mixin):
        mock_stream = MockAsyncStream([MockChunk("1")])

        assert not hasattr(mock_stream, "aclose")

        result = await mixin._maybe_overwrite_id(mock_stream, stream=True)

        assert hasattr(result, "aclose"), "Result MUST have aclose() for AsyncIterator"
        assert inspect.isasyncgen(result)
        assert isinstance(result, AsyncIterator)

    async def test_streaming_yields_all_chunks(self, mixin):
        chunks = [MockChunk("1", "a"), MockChunk("2", "b")]
        mock_stream = MockAsyncStream(chunks)

        result = await mixin._maybe_overwrite_id(mock_stream, stream=True)

        received = [c async for c in result]
        assert len(received) == 2
        assert received[0].content == "a"
        assert received[1].content == "b"

    async def test_non_streaming_returns_directly(self, mixin):
        mock_response = MagicMock()
        mock_response.id = "test-id"

        result = await mixin._maybe_overwrite_id(mock_response, stream=False)

        assert result is mock_response
        assert not inspect.isasyncgen(result)


class TestIdOverwriting:
    async def test_ids_overwritten_when_enabled(self):
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinTestImpl(config=config)
        mixin.overwrite_completion_id = True

        chunks = [MockChunk("orig-1"), MockChunk("orig-2")]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert all(c.id.startswith("cltsd-") for c in received)
        assert received[0].id == received[1].id  # Same ID for all chunks

    async def test_ids_preserved_when_disabled(self):
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinTestImpl(config=config)
        mixin.overwrite_completion_id = False

        chunks = [MockChunk("orig-1"), MockChunk("orig-2")]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert received[0].id == "orig-1"
        assert received[1].id == "orig-2"


class TestIssue5122UsageNormalization:
    """Regression tests for #5122: Gemini overcounts tokens in streaming mode.

    Gemini's OpenAI-compatible endpoint returns usage in every streaming chunk.
    The fix normalizes this so usage only appears on the final chunk.
    """

    async def test_usage_only_on_final_chunk_when_every_chunk_has_usage(self, mixin):
        """When a provider (e.g. Gemini) sends usage on every chunk, only the last chunk should have it."""
        chunks = [
            MockChunk("1", "Hello", usage=MockUsage(10, 1, 11)),
            MockChunk("1", " world", usage=MockUsage(10, 2, 12)),
            MockChunk("1", "!", usage=MockUsage(10, 3, 13)),
        ]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert len(received) == 3
        # Intermediate chunks should have no usage
        assert received[0].usage is None
        assert received[1].usage is None
        # Final chunk should have the last reported usage
        assert received[2].usage is not None
        assert received[2].usage.prompt_tokens == 10
        assert received[2].usage.completion_tokens == 3
        assert received[2].usage.total_tokens == 13

    async def test_usage_preserved_when_only_on_final_chunk(self, mixin):
        """When a compliant provider sends usage only on the final chunk, behavior is unchanged."""
        chunks = [
            MockChunk("1", "Hello"),
            MockChunk("1", " world"),
            MockChunk("1", "", usage=MockUsage(10, 5, 15)),
        ]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert len(received) == 3
        assert received[0].usage is None
        assert received[1].usage is None
        assert received[2].usage is not None
        assert received[2].usage.completion_tokens == 5

    async def test_no_usage_when_provider_sends_none(self, mixin):
        """When no chunks have usage, the final chunk should have None usage."""
        chunks = [
            MockChunk("1", "Hello"),
            MockChunk("1", " world"),
        ]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert len(received) == 2
        assert received[0].usage is None
        assert received[1].usage is None

    async def test_single_chunk_with_usage(self, mixin):
        """Single chunk with usage should preserve it."""
        chunks = [MockChunk("1", "Hi", usage=MockUsage(5, 1, 6))]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert len(received) == 1
        assert received[0].usage is not None
        assert received[0].usage.total_tokens == 6

    async def test_content_preserved_with_usage_normalization(self, mixin):
        """Content should be unaffected by usage normalization."""
        chunks = [
            MockChunk("1", "a", usage=MockUsage(10, 1, 11)),
            MockChunk("1", "b", usage=MockUsage(10, 2, 12)),
            MockChunk("1", "c", usage=MockUsage(10, 3, 13)),
        ]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert [c.content for c in received] == ["a", "b", "c"]

    async def test_empty_stream(self, mixin):
        """Empty stream should produce no chunks."""
        result = await mixin._maybe_overwrite_id(MockAsyncStream([]), stream=True)

        received = [c async for c in result]
        assert len(received) == 0
