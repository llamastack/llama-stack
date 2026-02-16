# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.providers.remote.inference.vertexai_native.config import VertexAINativeConfig
from llama_stack_api import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
)
from llama_stack_api.inference import RerankRequest


def build_chat_params(**kwargs: Any) -> OpenAIChatCompletionRequestWithExtraBody:
    payload: dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(kwargs)
    return OpenAIChatCompletionRequestWithExtraBody.model_validate(payload)


def _streaming_chunks(*items: Any) -> AsyncIterator[Any]:
    async def _gen() -> AsyncIterator[Any]:
        for item in items:
            yield item

    return _gen()


@pytest.fixture
def adapter_module():
    from llama_stack.providers.remote.inference.vertexai_native import vertexai_native

    return vertexai_native


@pytest.fixture
def adapter(adapter_module):
    return adapter_module.VertexAINativeInferenceAdapter(
        config=VertexAINativeConfig(project="cfg-project", location="us-central1")
    )


@pytest.fixture
def fake_genai(monkeypatch):
    import sys

    fake_client = MagicMock()
    fake_module = SimpleNamespace(Client=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)
    return SimpleNamespace(module=fake_module, client=fake_client)


async def test_initialize_is_noop(adapter):
    """Verify initialize() does not create a client (deferred to first request)."""
    await adapter.initialize()
    assert adapter._default_client is None


async def test_get_client_creates_and_caches_default_client(adapter, fake_genai):
    """Verify _get_client creates a client from config and caches it for reuse."""
    adapter.get_request_provider_data = MagicMock(return_value=None)

    client = await adapter._get_client()

    fake_genai.module.Client.assert_called_once_with(vertexai=True, project="cfg-project", location="us-central1")
    assert client is fake_genai.client
    assert adapter._default_client is fake_genai.client

    # Second call returns cached client without creating a new one
    fake_genai.module.Client.reset_mock()
    assert await adapter._get_client() is fake_genai.client
    fake_genai.module.Client.assert_not_called()


async def test_get_client_provider_data_overrides_project_and_location(adapter, fake_genai):
    """Verify per-request provider data overrides config defaults without caching."""
    adapter.get_request_provider_data = MagicMock(
        return_value=SimpleNamespace(vertex_project="request-project", vertex_location="us-east1")
    )

    client = await adapter._get_client()

    fake_genai.module.Client.assert_called_once_with(vertexai=True, project="request-project", location="us-east1")
    assert client is fake_genai.client
    assert adapter._default_client is None


@pytest.mark.parametrize(
    ("method_name", "payload", "error_message"),
    [
        (
            "openai_completion",
            OpenAICompletionRequestWithExtraBody(model="gemini-2.5-flash", prompt="hello"),
            "Text completion not supported. Use openai_chat_completion instead.",
        ),
        (
            "openai_embeddings",
            OpenAIEmbeddingsRequestWithExtraBody(model="gemini-embedding-001", input="hello"),
            "Embeddings not supported by vertexai_native provider.",
        ),
        (
            "rerank",
            RerankRequest(model="rerank-model", query="q", items=["a", "b"]),
            "Rerank not supported by vertexai_native provider.",
        ),
    ],
)
async def test_not_implemented_methods(adapter, method_name, payload, error_message):
    """Verify unsupported endpoints raise NotImplementedError with descriptive message."""
    with pytest.raises(NotImplementedError, match=error_message):
        await getattr(adapter, method_name)(payload)


async def test_check_model_availability_validates_against_known_models(adapter):
    """Verify known models are accepted and unknown models are rejected."""
    assert await adapter.check_model_availability("google/gemini-2.5-pro") is True
    assert await adapter.check_model_availability("google/gemini-2.0-flash") is True
    assert await adapter.check_model_availability("not-a-real-model") is False
    assert await adapter.check_model_availability("openai/gpt-4") is False


async def test_list_provider_model_ids_returns_known_models(adapter):
    """Verify provider returns known Gemini model IDs for auto-discovery."""
    model_ids = await adapter.list_provider_model_ids()
    assert "google/gemini-2.5-pro" in model_ids
    assert "google/gemini-2.5-flash" in model_ids
    assert "google/gemini-2.0-flash" in model_ids


@pytest.fixture
def mock_sdk(adapter, adapter_module, monkeypatch):
    """Provides a mock SDK client and patches all converter functions with sensible defaults."""
    mock_client = MagicMock()
    monkeypatch.setattr(adapter, "_get_client", AsyncMock(return_value=mock_client))
    monkeypatch.setattr(adapter_module, "logger", MagicMock())

    converted_contents = [SimpleNamespace()]
    generated_config = SimpleNamespace()

    monkeypatch.setattr(
        adapter_module.converters, "convert_messages", MagicMock(return_value=(None, converted_contents))
    )
    monkeypatch.setattr(adapter_module.converters, "convert_tools", MagicMock(return_value=(None, None)))
    monkeypatch.setattr(adapter_module.converters, "build_generate_config", MagicMock(return_value=generated_config))
    monkeypatch.setattr(adapter_module.converters, "collect_ignored_params", MagicMock(return_value=[]))

    return SimpleNamespace(
        client=mock_client,
        contents=converted_contents,
        config=generated_config,
        module=adapter_module,
    )


async def test_openai_chat_completion_non_streaming_delegates_to_sdk(adapter, mock_sdk, monkeypatch):
    """Verify non-streaming calls convert params, invoke the SDK, and return the converted response."""
    request_id = "req_nonstream"
    converted_tools = [SimpleNamespace()]
    converted_tool_config = SimpleNamespace()
    sdk_response = SimpleNamespace()
    converted_response = SimpleNamespace(id=request_id)

    params = build_chat_params(
        tools=[{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
        tool_choice="auto",
        stream=False,
    )

    mock_sdk.client.aio.models.generate_content = AsyncMock(return_value=sdk_response)
    monkeypatch.setattr(mock_sdk.module, "_new_request_id", MagicMock(return_value=request_id))
    monkeypatch.setattr(
        mock_sdk.module.converters, "convert_messages", MagicMock(return_value=("sys", mock_sdk.contents))
    )
    monkeypatch.setattr(
        mock_sdk.module.converters, "convert_tools", MagicMock(return_value=(converted_tools, converted_tool_config))
    )
    monkeypatch.setattr(mock_sdk.module.converters, "collect_ignored_params", MagicMock(return_value=["logprobs", "n"]))
    monkeypatch.setattr(mock_sdk.module.converters, "convert_response", MagicMock(return_value=converted_response))

    result = await adapter.openai_chat_completion(params)

    assert result is converted_response
    mock_sdk.module.converters.convert_messages.assert_called_once_with(params.messages)
    mock_sdk.module.converters.convert_tools.assert_called_once_with(params.tools, params.tool_choice)
    mock_sdk.module.converters.build_generate_config.assert_called_once_with(
        params, "sys", converted_tools, converted_tool_config
    )
    mock_sdk.client.aio.models.generate_content.assert_awaited_once_with(
        model=params.model, contents=mock_sdk.contents, config=mock_sdk.config
    )
    mock_sdk.module.converters.convert_response.assert_called_once_with(sdk_response, params.model, request_id)
    mock_sdk.module.logger.warning.assert_called_once_with("Ignoring unsupported params: ['logprobs', 'n']")


async def test_openai_chat_completion_streaming_returns_async_iterator(adapter, mock_sdk, monkeypatch):
    """Verify streaming calls return an async iterator of converted chunks."""
    request_id = "req_stream"
    params = build_chat_params(stream=True)
    chunk_1 = SimpleNamespace(text="a")
    chunk_2 = SimpleNamespace(text="b")

    mock_sdk.client.aio.models.generate_content_stream = AsyncMock(return_value=_streaming_chunks(chunk_1, chunk_2))
    monkeypatch.setattr(mock_sdk.module, "_new_request_id", MagicMock(return_value=request_id))
    monkeypatch.setattr(
        mock_sdk.module.converters,
        "convert_stream_chunk",
        MagicMock(side_effect=[SimpleNamespace(id="c1"), SimpleNamespace(id="c2")]),
    )

    result = await adapter.openai_chat_completion(params)
    chunks = [chunk async for chunk in result]

    assert [chunk.id for chunk in chunks] == ["c1", "c2"]
    mock_sdk.client.aio.models.generate_content_stream.assert_awaited_once_with(
        model=params.model, contents=mock_sdk.contents, config=mock_sdk.config
    )
    assert mock_sdk.module.converters.convert_stream_chunk.call_args_list[0].args == (
        chunk_1,
        params.model,
        request_id,
        0,
    )
    assert mock_sdk.module.converters.convert_stream_chunk.call_args_list[1].args == (
        chunk_2,
        params.model,
        request_id,
        1,
    )


async def test_openai_chat_completion_wraps_sdk_errors(adapter, mock_sdk, monkeypatch):
    """Verify SDK exceptions are wrapped with model context and chained via __cause__."""
    params = build_chat_params(stream=False)
    mock_sdk.client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("sdk boom"))
    monkeypatch.setattr(mock_sdk.module, "_new_request_id", MagicMock(return_value="req_err"))

    with pytest.raises(
        RuntimeError, match="Vertex AI native chat completion failed for model 'gemini-2.5-flash'"
    ) as exc:
        await adapter.openai_chat_completion(params)

    assert isinstance(exc.value.__cause__, RuntimeError)


async def test_streaming_cancellation_propagates_without_wrapping(adapter, mock_sdk, monkeypatch):
    """Verify asyncio.CancelledError bypasses the RuntimeError wrapper."""
    params = build_chat_params(stream=True)

    async def _cancel_gen() -> AsyncIterator[Any]:
        yield SimpleNamespace(text="first")
        raise asyncio.CancelledError

    mock_sdk.client.aio.models.generate_content_stream = AsyncMock(return_value=_cancel_gen())
    monkeypatch.setattr(mock_sdk.module, "_new_request_id", MagicMock(return_value="req_cancel"))
    monkeypatch.setattr(
        mock_sdk.module.converters,
        "convert_stream_chunk",
        MagicMock(return_value=SimpleNamespace(id="c1")),
    )

    result = await adapter.openai_chat_completion(params)
    with pytest.raises(asyncio.CancelledError):
        async for _ in result:
            pass


async def test_streaming_cleanup_calls_aclose(adapter, mock_sdk, monkeypatch):
    """Verify stream.aclose() is called after iteration completes."""
    params = build_chat_params(stream=True)
    chunk = SimpleNamespace(text="done")
    aclose_mock = AsyncMock()

    class FakeStream:
        """Async iterator with a trackable aclose method."""

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "_yielded"):
                self._yielded = True
                return chunk
            raise StopAsyncIteration

        async def aclose(self):
            await aclose_mock()

    mock_sdk.client.aio.models.generate_content_stream = AsyncMock(return_value=FakeStream())
    monkeypatch.setattr(mock_sdk.module, "_new_request_id", MagicMock(return_value="req_close"))
    monkeypatch.setattr(
        mock_sdk.module.converters,
        "convert_stream_chunk",
        MagicMock(return_value=SimpleNamespace(id="c1")),
    )

    result = await adapter.openai_chat_completion(params)
    async for _ in result:
        pass

    aclose_mock.assert_awaited_once()
