# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from llama_stack.providers.remote.inference.nvidia.config import NVIDIAConfig
from llama_stack.providers.remote.inference.nvidia.nvidia import NVIDIAInferenceAdapter


class MockResponse:
    def __init__(self, status=200, json_data=None, text_data="OK"):
        self.status = status
        self._json_data = json_data or {"rankings": []}
        self._text_data = text_data

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


class MockSession:
    def __init__(self, response):
        self.response = response
        self.post_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))

        class PostContext:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

        return PostContext(self.response)


def create_adapter(config=None, model_metadata=None):
    if config is None:
        config = NVIDIAConfig(api_key="test-key")

    adapter = NVIDIAInferenceAdapter(config)

    class MockModel:
        provider_resource_id = "test-model"
        metadata = model_metadata or {}

    adapter.model_store = AsyncMock()
    adapter.model_store.get_model = AsyncMock(return_value=MockModel())

    return adapter


async def test_rerank_basic_functionality():
    adapter = create_adapter()
    mock_response = MockResponse(json_data={"rankings": [{"index": 0, "logit": 0.5}]})
    mock_session = MockSession(mock_response)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await adapter.rerank(model="test-model", query="test query", items=["item1", "item2"])

    assert len(result.data) == 1
    assert result.data[0].index == 0
    assert result.data[0].relevance_score == 0.5

    url, kwargs = mock_session.post_calls[0]
    payload = kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["query"] == {"text": "test query"}
    assert payload["passages"] == [{"text": "item1"}, {"text": "item2"}]


async def test_missing_rankings_key():
    adapter = create_adapter()
    mock_session = MockSession(MockResponse(json_data={}))

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await adapter.rerank(model="test-model", query="q", items=["a"])

    assert len(result.data) == 0


async def test_hosted_with_endpoint():
    adapter = create_adapter(
        config=NVIDIAConfig(api_key="key"), model_metadata={"endpoint": "https://model.endpoint/rerank"}
    )
    mock_session = MockSession(MockResponse())

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await adapter.rerank(model="test-model", query="q", items=["a"])

    url, _ = mock_session.post_calls[0]
    assert url == "https://model.endpoint/rerank"


async def test_hosted_without_endpoint():
    adapter = create_adapter(
        config=NVIDIAConfig(api_key="key"),  # This creates hosted config (integrate.api.nvidia.com).
        model_metadata={},  # No "endpoint" key
    )
    mock_session = MockSession(MockResponse())

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await adapter.rerank(model="test-model", query="q", items=["a"])

    url, _ = mock_session.post_calls[0]
    assert "https://integrate.api.nvidia.com" in url


async def test_self_hosted_ignores_endpoint():
    adapter = create_adapter(
        config=NVIDIAConfig(url="http://localhost:8000", api_key=None),
        model_metadata={"endpoint": "https://model.endpoint/rerank"},  # This should be ignored.
    )
    mock_session = MockSession(MockResponse())

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await adapter.rerank(model="test-model", query="q", items=["a"])

    url, _ = mock_session.post_calls[0]
    assert "http://localhost:8000" in url
    assert "model.endpoint/rerank" not in url


async def test_max_num_results():
    adapter = create_adapter()
    rankings = [{"index": 0, "logit": 0.8}, {"index": 1, "logit": 0.6}]
    mock_session = MockSession(MockResponse(json_data={"rankings": rankings}))

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await adapter.rerank(model="test-model", query="q", items=["a", "b"], max_num_results=1)

    assert len(result.data) == 1
    assert result.data[0].index == 0
    assert result.data[0].relevance_score == 0.8


async def test_http_error():
    adapter = create_adapter()
    mock_session = MockSession(MockResponse(status=500, text_data="Server Error"))

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(ConnectionError, match="status 500.*Server Error"):
            await adapter.rerank(model="test-model", query="q", items=["a"])


async def test_client_error():
    adapter = create_adapter()
    mock_session = AsyncMock()
    mock_session.__aenter__.side_effect = aiohttp.ClientError("Network error")

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(ConnectionError, match="Failed to connect.*Network error"):
            await adapter.rerank(model="test-model", query="q", items=["a"])
