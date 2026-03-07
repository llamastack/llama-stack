# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import logging  # allow-direct-logging
import ssl
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr


async def _async_pager(items):
    """Yield items through an async iterator."""
    for item in items:
        yield item


def _make_fake_streaming_chunk(text: str = "chunk") -> SimpleNamespace:
    """Build a fake streaming chunk object."""
    part = SimpleNamespace(text=text, function_call=None, thought=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP", index=0, logprobs_result=None)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


from llama_stack.providers.remote.inference.vertexai.config import VertexAIConfig, VertexAIProviderDataValidator
from llama_stack.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter, _build_http_options
from llama_stack.providers.utils.inference.model_registry import NetworkConfig, ProxyConfig, TimeoutConfig, TLSConfig
from llama_stack_api import (
    Model,
    ModelType,
    OpenAIChatCompletionChunk,
    OpenAIEmbeddingsRequestWithExtraBody,
)
from llama_stack_api.inference.models import OpenAIChatCompletionRequestWithExtraBody


@pytest.fixture
def vertex_config() -> VertexAIConfig:
    """Handle vertex config."""
    return VertexAIConfig(project="test-project", location="global")


@pytest.fixture
def adapter(vertex_config: VertexAIConfig) -> VertexAIInferenceAdapter:
    """Handle adapter."""
    a = VertexAIInferenceAdapter(config=vertex_config)
    cast(Any, a).__provider_id__ = "vertexai"
    return a


@pytest.fixture
def patch_chat_completion_dependencies(monkeypatch):
    """Patch chat completion dependencies for the test."""

    def factory(
        adapter: VertexAIInferenceAdapter,
        *,
        capture_messages: bool = False,
        capture_tools: bool = False,
        capture_generation_kwargs: bool = False,
    ) -> dict[str, Any]:
        """Create an adapter with a mocked client."""
        fake_response = object()
        fake_completion = object()
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=fake_response)))
        )
        captured: dict[str, Any] = {"fake_completion": fake_completion}

        async def _provider_model_id(_: str) -> str:
            """Return a fixed provider model identifier."""
            return "gemini-2.5-flash"

        monkeypatch.setattr(adapter, "_get_provider_model_id", _provider_model_id)
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_model_name",
            lambda _: "gemini-2.5-flash",
        )

        if capture_generation_kwargs:

            def _build_generation_config(*_args, **kwargs):
                """Build eneration config."""
                captured["build_generation_config_kwargs"] = kwargs
                return object()

            monkeypatch.setattr(adapter, "_build_generation_config", _build_generation_config)
        else:
            monkeypatch.setattr(adapter, "_build_generation_config", lambda *_args, **_kwargs: object())

        if capture_messages:

            def _convert_messages(messages):
                """Convert essages."""
                captured["messages"] = messages
                return None, [{"role": "user", "parts": [{"text": "ok"}]}]

        else:

            def _convert_messages(messages):
                """Convert essages."""
                return None, [{"role": "user", "parts": [{"text": "ok"}]}]

        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_openai_messages_to_gemini",
            _convert_messages,
        )

        if capture_tools:

            def _convert_tools_capture(tools):
                """Convert ools capture."""
                captured["tools_passed"] = tools
                return None

            convert_tools = _convert_tools_capture
        else:

            def _convert_tools_passthrough(_tools):
                """Convert ools passthrough."""
                return None

            convert_tools = _convert_tools_passthrough

        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_openai_tools_to_gemini",
            convert_tools,
        )
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_gemini_response_to_openai",
            lambda response, model: fake_completion,
        )

        return captured

    return factory


class TestVertexAIAdapterInit:
    def test_init_sets_config_and_default_client(
        self, adapter: VertexAIInferenceAdapter, vertex_config: VertexAIConfig
    ):
        """Test that init sets config and default client."""
        assert adapter.config == vertex_config
        assert adapter._default_client is None

    async def test_initialize_sets_default_client(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Test that initialize sets default client."""
        client = object()

        monkeypatch.setattr(adapter, "_create_client", lambda **kwargs: client)

        await adapter.initialize()

        assert adapter._default_client is client

    async def test_initialize_failure_keeps_default_client_unset(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Test that initialize failure keeps default client unset."""

        def _raise(**kwargs):
            """Raise a runtime error for failure-path testing."""
            raise RuntimeError("boom")

        monkeypatch.setattr(adapter, "_create_client", _raise)

        await adapter.initialize()

        assert adapter._default_client is None


class TestVertexAIClientManagement:
    def test_create_client_with_access_token_uses_credentials(self, monkeypatch):
        """Test that create client with access token uses credentials."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("llama_stack.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="test-project", location="global"))
        client = adapter._create_client(
            project="test-project",
            location="global",
            access_token="token-123",
        )

        assert client is client_ctor.return_value
        kwargs = client_ctor.call_args.kwargs
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "test-project"
        assert kwargs["location"] == "global"
        assert kwargs["credentials"].token == "token-123"

    def test_get_client_uses_default_client(self, monkeypatch):
        """Test that get client uses default client."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        default_client = object()
        adapter._default_client = cast(Any, default_client)
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: None)

        assert adapter._get_client() is default_client

    def test_get_client_uses_provider_override_with_token(self, monkeypatch):
        """Test that get client uses provider override with token."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        override = VertexAIProviderDataValidator(
            vertex_project="override-project",
            vertex_location="us-central1",
            vertex_access_token=SecretStr("override-token"),
        )
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: override)

        create_client = MagicMock(return_value=object())
        monkeypatch.setattr(adapter, "_create_client", create_client)

        client = adapter._get_client()

        assert client is create_client.return_value
        create_client.assert_called_once_with(
            project="override-project",
            location="us-central1",
            access_token="override-token",
        )

    def test_get_client_project_override_reuses_configured_token(self, monkeypatch):
        """Test that get client project override reuses configured token."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", access_token=SecretStr("config-token")),
        )
        override = VertexAIProviderDataValidator(
            vertex_project="other-project",
            vertex_location=None,
            vertex_access_token=None,
        )
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: override)

        create_client = MagicMock(return_value=object())
        monkeypatch.setattr(adapter, "_create_client", create_client)

        client = adapter._get_client()

        assert client is create_client.return_value
        create_client.assert_called_once_with(
            project="other-project",
            location="l",
            access_token="config-token",
        )

    def test_get_client_raises_when_no_client_available(self, monkeypatch):
        """Test that get client raises when no client available."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: None)

        with pytest.raises(ValueError, match="Pass Vertex AI access token in the header"):
            adapter._get_client()


class TestVertexAIModelListing:
    async def test_list_provider_model_ids_filters_and_deduplicates(self, monkeypatch):
        """Test that list provider model ids filters and deduplicates."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        models = [
            SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-2.5-pro", supported_actions=[]),
            SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
            SimpleNamespace(name="", supported_actions=["generateContent"]),
        ]

        async def fake_list(**kwargs):
            """Handle fake list."""
            return _async_pager(models)

        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(list=fake_list)))
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        result = await adapter.list_provider_model_ids()

        assert "gemini-2.5-flash" in result
        assert "gemini-2.5-pro" in result
        assert "text-embedding-004" in result
        assert "" not in result

    @pytest.mark.parametrize(
        "index,expected_id,expected_type,expected_metadata",
        [
            pytest.param(0, "gemini-2.5-flash", ModelType.llm, None, id="flash_llm"),
            pytest.param(1, "gemini-2.5-pro", ModelType.llm, None, id="pro_llm"),
            pytest.param(
                2,
                "gemini-embedding-001",
                ModelType.embedding,
                {"embedding_dimension": 3072, "context_length": 2048},
                id="embedding",
            ),
        ],
    )
    async def test_list_models_returns_correct_attributes(
        self, monkeypatch, adapter, index, expected_id, expected_type, expected_metadata
    ):
        """Test that list models returns correct attributes."""
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-embedding-001"]),
        )

        models = await adapter.list_models()

        assert models is not None
        model = models[index]
        assert model.identifier == expected_id
        assert model.provider_resource_id == expected_id
        assert model.provider_id == "vertexai"
        assert model.model_type == expected_type
        if expected_metadata is not None:
            assert model.metadata == expected_metadata

    async def test_list_models_respects_allowed_models(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Test that list models respects allowed models."""
        monkeypatch.setattr(adapter.config, "allowed_models", ["gemini-2.5-flash"])
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["gemini-2.5-flash", "gemini-2.5-pro"]),
        )

        models = await adapter.list_models()

        assert models is not None
        assert len(models) == 1
        assert models[0].identifier == "gemini-2.5-flash"

    async def test_list_models_propagates_errors(self, monkeypatch):
        """Test that list models propagates errors."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(side_effect=RuntimeError("API unreachable")),
        )

        with pytest.raises(RuntimeError, match="API unreachable"):
            await adapter.list_models()

    async def test_should_refresh_models_returns_config_value(self):
        """Test that should refresh models returns config value."""
        adapter_default = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        assert await adapter_default.should_refresh_models() is False

        adapter_refresh = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", refresh_models=True),
        )
        assert await adapter_refresh.should_refresh_models() is True


class TestVertexAIModelAvailability:
    @pytest.mark.parametrize(
        "model,available_models,error,expected",
        [
            ("gemini-2.5-flash", ["gemini-2.5-flash"], None, True),
            ("nonexistent-model", ["gemini-2.5-flash"], None, False),
            ("anything", None, RuntimeError("offline"), True),
        ],
    )
    async def test_check_model_availability(self, monkeypatch, model, available_models, error, expected):
        """Test that check model availability."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        if error is not None:
            monkeypatch.setattr(adapter, "list_models", AsyncMock(side_effect=error))
        else:
            models = [
                Model(
                    provider_id="test-provider",
                    provider_resource_id=model_id,
                    identifier=model_id,
                    model_type=ModelType.llm,
                )
                for model_id in available_models
            ]

            async def mock_list_models():
                """Handle mock list models."""
                adapter._model_cache = {m.identifier: m for m in models}
                return models

            monkeypatch.setattr(adapter, "list_models", AsyncMock(side_effect=mock_list_models))

        assert await adapter.check_model_availability(model) is expected

    async def test_cache_hit_avoids_api_call(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Verify lazy load + cache reuse prevents redundant API calls."""
        mock = AsyncMock(return_value=["gemini-2.5-flash"])
        monkeypatch.setattr(adapter, "list_provider_model_ids", mock)

        # First call populates cache (triggers list_models → list_provider_model_ids)
        result1 = await adapter.check_model_availability("gemini-2.5-flash")
        assert result1 is True

        # Second call uses cache — list_provider_model_ids NOT called again
        result2 = await adapter.check_model_availability("gemini-2.5-flash")
        assert result2 is True

        assert mock.call_count == 1  # API called exactly once

    async def test_cached_unknown_model_returns_false_without_api_call(
        self, monkeypatch, adapter: VertexAIInferenceAdapter
    ):
        """Once cache is populated, unknown model returns False without hitting API again."""
        mock = AsyncMock(return_value=["gemini-2.5-flash"])
        monkeypatch.setattr(adapter, "list_provider_model_ids", mock)

        # Populate the cache via first check
        await adapter.check_model_availability("gemini-2.5-flash")

        # Unknown model — must return False using cache, NOT re-call API
        result = await adapter.check_model_availability("nonexistent-model")
        assert result is False
        assert mock.call_count == 1  # Still only called once

    async def test_list_models_populates_model_cache(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Verify list_models() populates _model_cache with Model objects."""
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["gemini-2.5-flash", "gemini-2.5-pro"]),
        )

        await adapter.list_models()

        assert "gemini-2.5-flash" in adapter._model_cache
        assert "gemini-2.5-pro" in adapter._model_cache
        assert len(adapter._model_cache) == 2

    async def test_list_models_clears_cache_on_refresh(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Verify cache is cleared and repopulated on list_models() call."""
        mock = AsyncMock(return_value=["model-a"])
        monkeypatch.setattr(adapter, "list_provider_model_ids", mock)

        await adapter.list_models()
        assert "model-a" in adapter._model_cache

        # Change what the API returns
        mock.return_value = ["model-b"]
        await adapter.list_models()

        # Cache should now only have model-b, not model-a
        assert "model-b" in adapter._model_cache
        assert "model-a" not in adapter._model_cache
        assert len(adapter._model_cache) == 1


class TestVertexAIAllowedModelsValidation:
    def test_validate_allowed_model_passes(self):
        """Test that validate allowed model passes."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=["gemini-2.5-flash"]),
        )
        adapter._validate_model_allowed("gemini-2.5-flash")

    def test_validate_disallowed_model_raises(self):
        """Test that validate disallowed model raises."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=["gemini-2.5-flash"]),
        )
        with pytest.raises(ValueError, match="not in the allowed models list"):
            adapter._validate_model_allowed("gemini-2.5-pro")

    def test_validate_no_allowed_models_passes_anything(self):
        """Test that validate no allowed models passes anything."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        adapter._validate_model_allowed("any-model-at-all")


class TestVertexAIUnsupportedOps:
    @pytest.mark.parametrize(
        "method_name,call_kwargs,error_pattern",
        [
            ("rerank", {"request": cast(Any, None)}, "rerank not yet implemented"),
        ],
    )
    async def test_unsupported_operations_raise(self, method_name, call_kwargs, error_pattern):
        """Verify that unimplemented operations raise NotImplementedError."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        with pytest.raises(NotImplementedError, match=error_pattern):
            await getattr(adapter, method_name)(**call_kwargs)


class TestOpenAIChatCompletionImagePreDownload:
    async def test_http_image_url_is_downloaded(self, monkeypatch, patch_chat_completion_dependencies):
        """Test that http image url is downloaded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        conversion_inputs = patch_chat_completion_dependencies(adapter, capture_messages=True)

        localize_image_content_mock = AsyncMock(return_value=(b"fake image bytes", "jpeg"))
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.localize_image_content",
            localize_image_content_mock,
        )

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpeg"}}],
                    }
                ],
            ),
        )

        await adapter.openai_chat_completion(params)

        localize_image_content_mock.assert_awaited_once_with("https://example.com/image.jpeg")
        converted_messages = conversion_inputs["messages"]
        image_url = converted_messages[0].content[0].image_url.url
        expected_data = base64.b64encode(b"fake image bytes").decode("utf-8")
        assert image_url == f"data:image/jpeg;base64,{expected_data}"

    async def test_data_uri_not_re_downloaded(self, monkeypatch, patch_chat_completion_dependencies):
        """Test that data uri not re downloaded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        conversion_inputs = patch_chat_completion_dependencies(adapter, capture_messages=True)

        localize_image_content_mock = AsyncMock()
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.localize_image_content",
            localize_image_content_mock,
        )

        data_uri = "data:image/png;base64,ZmFrZQ=="
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
                    }
                ],
            ),
        )

        await adapter.openai_chat_completion(params)

        localize_image_content_mock.assert_not_called()
        converted_messages = conversion_inputs["messages"]
        assert converted_messages[0].content[0].image_url.url == data_uri

    async def test_failed_download_raises_value_error(self, monkeypatch, patch_chat_completion_dependencies):
        """Test that failed download raises value error."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        localize_image_content_mock = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.localize_image_content",
            localize_image_content_mock,
        )

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}],
                    }
                ],
            ),
        )

        with pytest.raises(ValueError, match="Failed to localize image content"):
            await adapter.openai_chat_completion(params)


class TestCollectSamplingParams:
    """Test VertexAIInferenceAdapter._collect_sampling_params() for new sampling parameters."""

    @pytest.mark.parametrize(
        "param_name,param_value,expected_key,expected_value",
        [
            pytest.param("frequency_penalty", 0.5, "frequency_penalty", 0.5, id="frequency_penalty"),
            pytest.param("presence_penalty", 0.5, "presence_penalty", 0.5, id="presence_penalty"),
            pytest.param("seed", 42, "seed", 42, id="seed"),
        ],
    )
    def test_single_sampling_param_forwarded(self, param_name, param_value, expected_key, expected_value):
        """Test that single sampling param forwarded."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            **{param_name: param_value},
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert expected_key in result
        assert result[expected_key] == expected_value

    def test_all_three_params_combined(self):
        """Test that all three new params are collected together."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            frequency_penalty=0.3,
            presence_penalty=0.4,
            seed=123,
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert result["frequency_penalty"] == 0.3
        assert result["presence_penalty"] == 0.4
        assert result["seed"] == 123

    def test_none_values_are_excluded(self):
        """Test that None values are not included in result."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert "seed" not in result

    def test_zero_values_are_forwarded(self):
        """Test that zero values are forwarded (not dropped as falsy)."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=0,
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert "frequency_penalty" in result
        assert result["frequency_penalty"] == 0.0
        assert "presence_penalty" in result
        assert result["presence_penalty"] == 0.0
        assert "seed" in result
        assert result["seed"] == 0

    def test_existing_params_unaffected(self):
        """Test that existing params (temperature, top_p) still work alongside new params."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            seed=99,
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["frequency_penalty"] == 0.2
        assert result["presence_penalty"] == 0.3
        assert result["seed"] == 99

    @pytest.mark.parametrize(
        "input_param,input_value,expected_key,expected_value",
        [
            pytest.param("logprobs", True, "response_logprobs", True, id="logprobs_to_response_logprobs"),
            pytest.param("top_logprobs", 5, "logprobs", 5, id="top_logprobs_to_logprobs"),
        ],
    )
    def test_logprobs_mapping(self, input_param, input_value, expected_key, expected_value):
        """Test that logprobs mapping."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            **{input_param: input_value},
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert expected_key in result
        assert result[expected_key] == expected_value

    def test_logprobs_none_excluded(self):
        """Test that logprobs and top_logprobs keys are absent when params are None."""
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
        )
        result = VertexAIInferenceAdapter._collect_sampling_params(params)
        assert "response_logprobs" not in result
        assert "logprobs" not in result


class TestVertexAINetworkConfig:
    """Tests for _build_http_options() conversion from NetworkConfig to HttpOptions."""

    def test_build_http_options_none_returns_none(self):
        """Test that build http options none returns none."""
        assert _build_http_options(None) is None

    def test_build_http_options_empty_config_returns_none(self):
        # All fields are None by default — nothing to configure
        """Test that build http options empty config returns none."""
        assert _build_http_options(NetworkConfig()) is None

    def test_build_http_options_headers_only(self):
        """Test that build http options headers only."""
        result = _build_http_options(NetworkConfig(headers={"X-Custom": "value"}))
        assert result is not None
        assert result.headers == {"X-Custom": "value"}

    def test_build_http_options_float_timeout_converts_to_ms(self):
        """Test that build http options float timeout converts to ms."""
        result = _build_http_options(NetworkConfig(timeout=30.0))
        assert result is not None
        assert result.timeout == 30000

    def test_build_http_options_timeout_config_read_wins(self):
        """Test that build http options timeout config read wins."""
        result = _build_http_options(NetworkConfig(timeout=TimeoutConfig(read=60.0)))
        assert result is not None
        assert result.timeout == 60000

    def test_build_http_options_timeout_config_connect_fallback(self):
        # When read is None, fall back to connect
        """Test that build http options timeout config connect fallback."""
        result = _build_http_options(NetworkConfig(timeout=TimeoutConfig(connect=5.0)))
        assert result is not None
        assert result.timeout == 5000

    def test_build_http_options_timeout_config_both_uses_read(self):
        # read takes priority over connect
        """Test that build http options timeout config both uses read."""
        result = _build_http_options(NetworkConfig(timeout=TimeoutConfig(connect=5.0, read=30.0)))
        assert result is not None
        assert result.timeout == 30000

    def test_build_http_options_timeout_config_both_none_returns_none(self):
        # If both connect and read are None, no timeout added → no other fields → None
        """Test that build http options timeout config both none returns none."""
        assert _build_http_options(NetworkConfig(timeout=TimeoutConfig())) is None

    def test_build_http_options_tls_verify_false_produces_ssl_context(self):
        # verify=False should produce an explicit ssl.SSLContext with CERT_NONE so it's
        # truthy and won't be replaced by the SDK's _ensure_httpx_ssl_ctx().
        """Test that build http options tls verify false produces ssl context."""
        result = _build_http_options(NetworkConfig(tls=TLSConfig(verify=False)))
        assert result is not None
        assert hasattr(result, "httpx_async_client")
        client = result.httpx_async_client
        assert client is not None
        # The verify kwarg should be an SSLContext (not bare False) — access via internal pool
        pool = client._transport._pool  # type: ignore[union-attr]  # httpx.AsyncHTTPTransport has _pool
        assert isinstance(pool._ssl_context, ssl.SSLContext)
        assert pool._ssl_context.verify_mode == ssl.CERT_NONE

    def test_build_http_options_tls_verify_true_still_sets_httpx_client(self):
        # Even verify=True triggers httpx_async_client so we can set follow_redirects
        """Test that build http options tls verify true still sets httpx client."""
        result = _build_http_options(NetworkConfig(tls=TLSConfig(verify=True)))
        assert result is not None
        assert hasattr(result, "httpx_async_client")
        assert result.httpx_async_client is not None

    def test_build_http_options_tls_verify_false_httpx_client_has_follow_redirects(self):
        """Test that build http options tls verify false httpx client has follow redirects."""
        result = _build_http_options(NetworkConfig(tls=TLSConfig(verify=False)))
        assert result is not None
        client = result.httpx_async_client
        assert client is not None
        assert client.follow_redirects is True or getattr(client, "_follow_redirects", True) is True

    def test_build_http_options_proxy_url_produces_httpx_client(self):
        """Test that build http options proxy url produces httpx client."""
        result = _build_http_options(
            NetworkConfig(proxy=ProxyConfig(url="http://proxy.example.com:8080"))  # type: ignore[arg-type]  # Pydantic coerces str→HttpUrl
        )
        assert result is not None
        assert hasattr(result, "httpx_async_client")
        assert result.httpx_async_client is not None

    def test_build_http_options_no_proxy_logs_warning(self, caplog):
        """Test that build http options no proxy logs warning."""
        with caplog.at_level(logging.WARNING):
            _build_http_options(NetworkConfig(proxy=ProxyConfig(no_proxy=["localhost"])))
        assert any("no_proxy" in record.message for record in caplog.records)

    def test_build_http_options_combined_all_fields(self):
        """Test that build http options combined all fields."""
        result = _build_http_options(
            NetworkConfig(
                headers={"X-Custom": "val"},
                timeout=30.0,
                tls=TLSConfig(verify=False),
                proxy=ProxyConfig(url="http://proxy.example.com:8080"),  # type: ignore[arg-type]  # Pydantic coerces str→HttpUrl
            )
        )
        assert result is not None
        assert result.headers == {"X-Custom": "val"}
        assert result.timeout == 30000
        assert hasattr(result, "httpx_async_client")
        assert result.httpx_async_client is not None


class TestVertexAIClientWithNetworkConfig:
    """Tests that client factory methods thread http_options into Client() calls."""

    def test_create_client_with_token_passes_http_options(self, monkeypatch):
        """Test that create client with token passes http options."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("llama_stack.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        http_opts = _build_http_options(NetworkConfig(headers={"X-Test": "1"}))
        adapter._http_options = http_opts

        adapter._create_client(project="p", location="l", access_token="tok")

        kwargs = client_ctor.call_args.kwargs
        assert kwargs.get("http_options") is http_opts

    def test_create_adc_client_passes_http_options(self, monkeypatch):
        """Test that create adc client passes http options."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("llama_stack.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        http_opts = _build_http_options(NetworkConfig(headers={"X-Test": "1"}))
        adapter._http_options = http_opts

        adapter._create_adc_client(project="p", location="l")

        kwargs = client_ctor.call_args.kwargs
        assert kwargs.get("http_options") is http_opts

    def test_create_client_no_network_config_no_http_options(self, monkeypatch):
        """Test that create client no network config no http options."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("llama_stack.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        # _http_options is None by default (no network config)
        assert adapter._http_options is None

        adapter._create_client(project="p", location="l", access_token="tok")

        kwargs = client_ctor.call_args.kwargs
        # http_options should not be in kwargs when network config is not set
        assert "http_options" not in kwargs

    async def test_initialize_builds_http_options_from_config(self, monkeypatch):
        """Test that initialize builds http options from config."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(
                project="p",
                location="l",
                network=NetworkConfig(headers={"X-Custom": "val"}),
            )
        )
        mock_client = object()
        monkeypatch.setattr(adapter, "_create_client", lambda **kw: mock_client)

        await adapter.initialize()

        # After initialize(), _http_options should be populated from config.network
        assert adapter._http_options is not None
        assert adapter._http_options.headers == {"X-Custom": "val"}

    async def test_shutdown_closes_managed_httpx_async_client(self, monkeypatch):
        """Test that shutdown closes managed httpx async client."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(
                project="p",
                location="l",
                network=NetworkConfig(tls=TLSConfig(verify=False)),
            )
        )
        adapter._http_options = _build_http_options(adapter.config.network)
        assert adapter._http_options is not None
        client = adapter._http_options.httpx_async_client
        assert client is not None
        aclose_mock = AsyncMock()
        monkeypatch.setattr(client, "aclose", aclose_mock)

        await adapter.shutdown()

        aclose_mock.assert_awaited_once()
        assert adapter._http_options is None
        assert adapter._default_client is None

    async def test_initialize_closes_existing_managed_httpx_async_client(self, monkeypatch):
        """Test that initialize closes existing managed httpx async client."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(
                project="p",
                location="l",
                network=NetworkConfig(headers={"X-New": "1"}),
            )
        )
        adapter._http_options = _build_http_options(NetworkConfig(tls=TLSConfig(verify=False)))
        assert adapter._http_options is not None
        old_client = adapter._http_options.httpx_async_client
        assert old_client is not None
        aclose_mock = AsyncMock()
        monkeypatch.setattr(old_client, "aclose", aclose_mock)
        monkeypatch.setattr(adapter, "_create_client", lambda **kw: object())

        await adapter.initialize()

        aclose_mock.assert_awaited_once()
        assert adapter._http_options is not None
        assert adapter._http_options.headers == {"X-New": "1"}


class TestBuildThinkingConfig:
    def test_none_returns_none(self):
        """Test that none returns none."""
        result = VertexAIInferenceAdapter._build_thinking_config(None)
        assert result is None

    def test_none_string_uses_budget_zero(self):
        """Test that none string uses budget zero."""
        result = VertexAIInferenceAdapter._build_thinking_config("none")
        assert result is not None
        assert result.thinking_budget == 0

    @pytest.mark.parametrize(
        "effort,expected_level",
        [
            ("minimal", "MINIMAL"),
            ("low", "LOW"),
            ("medium", "MEDIUM"),
            ("high", "HIGH"),
            ("xhigh", "HIGH"),
        ],
    )
    def test_effort_maps_to_thinking_level(self, effort: str, expected_level: str):
        """Test that effort maps to thinking level."""
        result = VertexAIInferenceAdapter._build_thinking_config(effort)
        assert result is not None
        assert result.thinking_level == expected_level

    def test_model_extra_overrides_thinking_config(self):
        """Test that model extra overrides thinking config."""
        override = {"thinking_budget": 9999}
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            reasoning_effort="low",
        )
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                **params.model_dump(exclude_none=True),
                "thinking_config": override,
            }
        )
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="proj", location="us-central1"),
        )
        config = adapter._build_generation_config(
            params,
            system_instruction=None,
            tools_input=None,
        )
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 9999


class TestStreamChatCompletion:
    """Test _stream_chat_completion() stream_options include_usage behavior."""

    def _make_fake_chunk(
        self,
        text: str = "hello",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int = 15,
    ) -> Any:
        """Build a fake chunk with usage metadata."""
        usage = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=total_tokens,
        )
        chunk = _make_fake_streaming_chunk(text)
        candidate = chunk.candidates[0]
        return SimpleNamespace(candidates=[candidate], usage_metadata=usage)

    async def test_stream_with_include_usage_emits_final_chunk(self, monkeypatch):
        """When stream_options include_usage=True, a final usage-only chunk is emitted."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1 = self._make_fake_chunk("Hello", prompt_tokens=10, completion_tokens=3, total_tokens=13)
        chunk2 = self._make_fake_chunk(" world", prompt_tokens=10, completion_tokens=5, total_tokens=15)

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield chunk1
            yield chunk2

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        stream = await adapter._stream_chat_completion(
            client=cast(Any, fake_client),
            provider_model_id="gemini-2.0-flash",
            contents=[],
            config=cast(Any, SimpleNamespace()),
            model="gemini-2.0-flash",
            stream_options={"include_usage": True},
        )

        chunks = [chunk async for chunk in stream]

        # 2 content chunks + 1 final usage-only chunk
        assert len(chunks) == 3

        # Final chunk has empty choices and usage populated
        final = chunks[-1]
        assert final.choices == []
        assert final.usage is not None
        assert final.usage.completion_tokens == 5
        assert final.usage.total_tokens == 15

    async def test_stream_without_stream_options_no_extra_chunk(self, monkeypatch):
        """When stream_options is None, only original content chunks are yielded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1 = self._make_fake_chunk("Hello")
        chunk2 = self._make_fake_chunk(" world")

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield chunk1
            yield chunk2

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        stream = await adapter._stream_chat_completion(
            client=cast(Any, fake_client),
            provider_model_id="gemini-2.0-flash",
            contents=[],
            config=cast(Any, SimpleNamespace()),
            model="gemini-2.0-flash",
            stream_options=None,
        )

        chunks = [chunk async for chunk in stream]

        # Exactly 2 chunks — no extra usage chunk
        assert len(chunks) == 2

    async def test_stream_with_include_usage_false_no_extra_chunk(self, monkeypatch):
        """When stream_options include_usage=False, no extra usage chunk is emitted."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1 = self._make_fake_chunk("Hi")

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield chunk1

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        stream = await adapter._stream_chat_completion(
            client=cast(Any, fake_client),
            provider_model_id="gemini-2.0-flash",
            contents=[],
            config=cast(Any, SimpleNamespace()),
            model="gemini-2.0-flash",
            stream_options={"include_usage": False},
        )

        chunks = [chunk async for chunk in stream]

        # Exactly 1 chunk — include_usage=False means no extra chunk
        assert len(chunks) == 1


@pytest.fixture
def make_adapter_with_mock_embed(monkeypatch):
    """Create adapter with mock embed."""

    def factory(
        embeddings: list[list[float]], capture: dict | None = None, usage_metadata: SimpleNamespace | None = None
    ):
        """Create an adapter with a mocked client."""
        a = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        cast(Any, a).__provider_id__ = "vertexai"

        fake_embedding_objects = [SimpleNamespace(value=vec) for vec in embeddings]
        fake_response = SimpleNamespace(embeddings=fake_embedding_objects)
        if usage_metadata is not None:
            fake_response.usage_metadata = usage_metadata

        embed_mock = AsyncMock(return_value=fake_response)

        if capture is not None:

            async def capturing_embed(*args, **kwargs):
                """Handle capturing embed."""
                capture.update(kwargs)
                return fake_response

            embed_mock = capturing_embed

        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_mock)))
        monkeypatch.setattr(a, "_get_client", lambda: fake_client)
        return a, embed_mock

    return factory


class TestVertexAIEmbeddings:
    """Tests for openai_embeddings() implementation."""

    async def test_single_string_returns_one_embedding(self, make_adapter_with_mock_embed):
        """Test that single string returns one embedding."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2, 0.3]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello world")
        result = await adapter.openai_embeddings(params)

        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[0].index == 0
        assert result.data[0].object == "embedding"
        assert result.model == "text-embedding-004"

    async def test_batch_strings_returns_multiple_embeddings(self, make_adapter_with_mock_embed):
        """Test that batch strings returns multiple embeddings."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input=["a", "b", "c"])
        result = await adapter.openai_embeddings(params)

        assert len(result.data) == 3
        assert result.data[0].embedding == [0.1, 0.2]
        assert result.data[1].embedding == [0.3, 0.4]
        assert result.data[2].embedding == [0.5, 0.6]
        for i, item in enumerate(result.data):
            assert item.index == i

    async def test_dimensions_forwarded_as_output_dimensionality(self, make_adapter_with_mock_embed):
        """Test that dimensions forwarded as output dimensionality."""
        capture: dict = {}
        adapter, _ = make_adapter_with_mock_embed([[0.1]], capture)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello", dimensions=512)
        await adapter.openai_embeddings(params)

        assert "config" in capture
        assert capture["config"].output_dimensionality == 512

    async def test_user_forwarded_as_labels(self, make_adapter_with_mock_embed):
        """Test that user forwarded as labels."""
        capture: dict = {}
        adapter, _ = make_adapter_with_mock_embed([[0.1]], capture)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello", user="alice")
        await adapter.openai_embeddings(params)

        assert "config" in capture
        assert capture["config"].labels == {"user": "alice"}

    async def test_base64_encoding_format(self, make_adapter_with_mock_embed):
        """Test that base64 encoding format."""
        import struct as _struct

        values = [0.1, 0.2, 0.3]
        adapter, _ = make_adapter_with_mock_embed([values])

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="text-embedding-004", input="hello", encoding_format="base64"
        )
        result = await adapter.openai_embeddings(params)

        embedding = result.data[0].embedding
        assert isinstance(embedding, str)
        decoded_bytes = base64.b64decode(embedding)
        decoded_floats = list(_struct.unpack(f"{len(values)}f", decoded_bytes))
        assert len(decoded_floats) == len(values)
        for orig, dec in zip(values, decoded_floats, strict=False):
            assert abs(orig - dec) < 1e-6

    async def test_token_array_input_raises_value_error(self, make_adapter_with_mock_embed):
        """Test that token array input raises value error."""
        adapter, _ = make_adapter_with_mock_embed([[0.1]])

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="text-embedding-004",
            input=cast(Any, [1, 2, 3]),  # token array, not text
        )
        with pytest.raises((ValueError, AttributeError)):
            await adapter.openai_embeddings(params)

    async def test_usage_returns_zeros(self, make_adapter_with_mock_embed):
        """Test that usage returns zeros."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    async def test_no_config_when_no_options(self, make_adapter_with_mock_embed):
        """When no dimensions or user are set, config should be None."""
        capture: dict = {}
        adapter, _ = make_adapter_with_mock_embed([[0.1]], capture)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="hello")
        await adapter.openai_embeddings(params)

        assert capture.get("config") is None

    async def test_embedding_usage_with_real_tokens(self, make_adapter_with_mock_embed):
        """When response has usage_metadata, usage shows real token counts."""
        usage_metadata = SimpleNamespace(prompt_token_count=10, total_token_count=15)
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]], usage_metadata=usage_metadata)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 15

    async def test_embedding_usage_fallback_when_no_metadata(self, make_adapter_with_mock_embed):
        """When response has no usage_metadata, usage falls back to zeros."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    async def test_embedding_usage_fallback_when_metadata_missing_fields(self, make_adapter_with_mock_embed):
        """When usage_metadata exists but fields are missing, fall back to zeros."""
        usage_metadata = SimpleNamespace()  # No token count fields
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]], usage_metadata=usage_metadata)

        params = OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-004", input="test")
        result = await adapter.openai_embeddings(params)

        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0


class TestDroppedParameterWarnings:
    """Test that unsupported parameters generate appropriate warning/debug logs."""

    @pytest.mark.parametrize(
        "param_name,param_value,log_level,expected_text",
        [
            pytest.param("logit_bias", {"50256": -100.0}, logging.WARNING, "logit_bias", id="logit_bias"),
            pytest.param(
                "service_tier",
                "__service_tier_default__",
                logging.WARNING,
                "service_tier",
                id="service_tier",
            ),
            pytest.param("prompt_cache_key", "mykey", logging.WARNING, "prompt_cache_key", id="prompt_cache_key"),
            pytest.param("user", "test-user", logging.DEBUG, "user", id="user"),
            pytest.param("safety_identifier", "test-id", logging.DEBUG, "safety_identifier", id="safety_identifier"),
        ],
    )
    async def test_unsupported_param_logged(
        self, monkeypatch, caplog, patch_chat_completion_dependencies, param_name, param_value, log_level, expected_text
    ):
        """Test that unsupported param logged."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        if param_name == "service_tier":
            from llama_stack_api.inference import ServiceTier

            param_value = ServiceTier.default

        payload: dict[str, Any] = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            param_name: param_value,
        }
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(payload)

        with caplog.at_level(logging.DEBUG, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert any(expected_text in r.message for r in caplog.records if r.levelno == log_level)

    @pytest.mark.parametrize(
        "value,should_warn",
        [
            pytest.param(False, True, id="false_warns"),
            pytest.param(True, False, id="true_no_warn"),
        ],
    )
    async def test_parallel_tool_calls_warning(
        self, monkeypatch, caplog, patch_chat_completion_dependencies, value, should_warn
    ):
        """Test that parallel tool calls warning."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        patch_chat_completion_dependencies(adapter)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            parallel_tool_calls=value,
        )

        with caplog.at_level(logging.WARNING):
            await adapter.openai_chat_completion(params)

        found = any("parallel tool calls" in r.message for r in caplog.records if r.levelno == logging.WARNING)
        assert found == should_warn


class TestTelemetryStreamOptions:
    """Test that telemetry stream options are injected when appropriate."""

    def _patch_stream_chat_completion(self, monkeypatch, adapter: VertexAIInferenceAdapter) -> dict[str, Any]:
        """Patch dependencies for streaming chat completion."""
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=None)))
        )
        stream_call_args: dict[str, Any] = {}

        async def _provider_model_id(_: str) -> str:
            """Return a fixed provider model identifier."""
            return "gemini-2.5-flash"

        async def _stream_chat_completion(client, model_id, contents, config, model, stream_options=None):
            """Handle stream chat completion."""
            stream_call_args["stream_options"] = stream_options
            return _async_pager([])

        monkeypatch.setattr(adapter, "_get_provider_model_id", _provider_model_id)
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_model_name",
            lambda _: "gemini-2.5-flash",
        )
        monkeypatch.setattr(adapter, "_build_generation_config", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_openai_messages_to_gemini",
            lambda messages: (None, [{"role": "user", "parts": [{"text": "ok"}]}]),
        )
        monkeypatch.setattr(
            "llama_stack.providers.remote.inference.vertexai.vertexai.converters.convert_openai_tools_to_gemini",
            lambda _tools: None,
        )
        monkeypatch.setattr(adapter, "_stream_chat_completion", _stream_chat_completion)

        return stream_call_args

    async def test_stream_options_injected_when_telemetry_active(self, monkeypatch):
        """When telemetry span is recording, stream_options should include include_usage=True."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        stream_call_args = self._patch_stream_chat_completion(monkeypatch, adapter)

        # Mock opentelemetry to return a recording span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span
        monkeypatch.setattr("opentelemetry.trace", mock_trace)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            stream=True,
        )

        # Consume the async generator
        result = await adapter.openai_chat_completion(params)
        async for _ in cast(AsyncIterator[OpenAIChatCompletionChunk], result):
            pass

        # Verify stream_options was passed with include_usage=True
        assert stream_call_args.get("stream_options") is not None
        assert stream_call_args["stream_options"].get("include_usage") is True

    async def test_stream_options_not_injected_when_telemetry_inactive(self, monkeypatch):
        """When telemetry span is not recording, stream_options should be None."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        stream_call_args = self._patch_stream_chat_completion(monkeypatch, adapter)

        # Mock opentelemetry to return a non-recording span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span
        monkeypatch.setattr("opentelemetry.trace", mock_trace)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            messages=cast(Any, [{"role": "user", "content": "hi"}]),
            stream=True,
        )

        # Consume the async generator
        result = await adapter.openai_chat_completion(params)
        async for _ in cast(AsyncIterator[OpenAIChatCompletionChunk], result):
            pass

        # Verify stream_options was not modified
        assert stream_call_args.get("stream_options") is None


class TestEmbeddingsModelExtra:
    """Test that model_extra parameter generates debug logs in embeddings."""

    async def test_model_extra_debug_log(self, make_adapter_with_mock_embed, caplog):
        """model_extra parameter should generate a DEBUG log."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody.model_validate(
            {
                "model": "text-embedding-004",
                "input": "hello",
                "custom_param": "value",
                "another_param": 123,
            }
        )

        with caplog.at_level(logging.DEBUG, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_embeddings(params)

        # Should have a DEBUG log mentioning model_extra or extra body parameters
        assert any(
            ("model_extra" in record.message or "extra body parameters" in record.message)
            for record in caplog.records
            if record.levelno == logging.DEBUG
        )

    async def test_no_model_extra_no_debug_log(self, make_adapter_with_mock_embed, caplog):
        """When model_extra is empty, no debug log should be generated."""
        adapter, _ = make_adapter_with_mock_embed([[0.1, 0.2]])

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="text-embedding-004",
            input="hello",
        )

        with caplog.at_level(logging.DEBUG, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_embeddings(params)

        # Should NOT have a debug log about model_extra
        assert not any(
            ("model_extra" in record.message or "extra body parameters" in record.message)
            for record in caplog.records
            if record.levelno == logging.DEBUG
        )


class TestDeprecatedFunctionCalling:
    async def test_functions_converted_to_tools_when_tools_absent(
        self, monkeypatch, caplog, patch_chat_completion_dependencies
    ):
        """Test that functions converted to tools when tools absent."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        functions = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}]
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "functions": functions,
            }
        )

        with caplog.at_level(logging.WARNING, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert any("functions" in record.message and "deprecated" in record.message for record in caplog.records)
        tools_passed = captured["tools_passed"]
        assert tools_passed is not None
        assert len(tools_passed) == 1
        assert tools_passed[0]["type"] == "function"
        assert tools_passed[0]["function"]["name"] == "get_weather"

    async def test_tools_takes_priority_over_functions(self, monkeypatch, caplog, patch_chat_completion_dependencies):
        """Test that tools takes priority over functions."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        modern_tools = [{"type": "function", "function": {"name": "modern_tool", "description": "Modern"}}]
        functions = [{"name": "legacy_func", "description": "Legacy"}]
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": modern_tools,
                "functions": functions,
            }
        )

        with caplog.at_level(logging.WARNING, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert not any("functions" in record.message and "deprecated" in record.message for record in caplog.records)
        tools_passed = captured["tools_passed"]
        assert tools_passed is not None
        assert len(tools_passed) == 1
        assert tools_passed[0]["function"]["name"] == "modern_tool"

    async def test_function_call_converted_to_tool_choice(
        self, monkeypatch, caplog, patch_chat_completion_dependencies
    ):
        """Test that function call converted to tool choice."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "function_call": {"name": "get_weather"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert any("function_call" in record.message and "deprecated" in record.message for record in caplog.records)
        kwargs = captured["build_generation_config_kwargs"]
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    async def test_tool_choice_takes_priority_over_function_call(
        self, monkeypatch, caplog, patch_chat_completion_dependencies
    ):
        """Test that tool choice takes priority over function call."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        captured = patch_chat_completion_dependencies(
            adapter,
            capture_tools=True,
            capture_generation_kwargs=True,
        )

        params = OpenAIChatCompletionRequestWithExtraBody.model_validate(
            {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "required",
                "function_call": {"name": "get_weather"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="llama_stack.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_chat_completion(params)

        assert not any(
            "function_call" in record.message and "deprecated" in record.message for record in caplog.records
        )
        kwargs = captured["build_generation_config_kwargs"]
        assert kwargs["tool_choice"] == "required"
