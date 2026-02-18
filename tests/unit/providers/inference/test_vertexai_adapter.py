# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr


async def _async_pager(items):
    for item in items:
        yield item


def _install_google_shims() -> None:
    """Install mock google-genai modules so tests run without the real SDK.

    IMPORTANT: This shim is tightly coupled to production imports.  If you add
    a new ``from google.genai import ...`` in the adapter code, you must update
    the corresponding mock module here.

    Shims are installed unconditionally (overwriting any previously-imported
    real SDK modules) so that tests are deterministic regardless of import
    order or whether google-genai is installed.
    """
    google_module = sys.modules.get("google")
    if google_module is None:
        google_module = ModuleType("google")
        google_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["google"] = google_module

    genai_module = ModuleType("google.genai")

    class MockClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = SimpleNamespace(list=lambda: [])
            self.aio = SimpleNamespace(models=SimpleNamespace())

    cast(Any, genai_module).Client = MockClient
    sys.modules["google.genai"] = genai_module
    cast(Any, google_module).genai = genai_module

    genai_types_module = ModuleType("google.genai.types")

    class FunctionCallingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ToolConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Tool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ListModelsConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    cast(Any, genai_types_module).FunctionCallingConfig = FunctionCallingConfig
    cast(Any, genai_types_module).ToolConfig = ToolConfig
    cast(Any, genai_types_module).Tool = Tool
    cast(Any, genai_types_module).GenerateContentConfig = GenerateContentConfig
    cast(Any, genai_types_module).ListModelsConfig = ListModelsConfig

    sys.modules["google.genai.types"] = genai_types_module
    cast(Any, genai_module).types = genai_types_module

    oauth2_module = ModuleType("google.oauth2")
    sys.modules["google.oauth2"] = oauth2_module
    cast(Any, google_module).oauth2 = oauth2_module

    credentials_module = ModuleType("google.oauth2.credentials")

    class Credentials:
        def __init__(self, token: str):
            self.token = token

    cast(Any, credentials_module).Credentials = Credentials
    sys.modules["google.oauth2.credentials"] = credentials_module
    cast(Any, oauth2_module).credentials = credentials_module


_install_google_shims()

from llama_stack.providers.remote.inference.vertexai.config import VertexAIConfig, VertexAIProviderDataValidator
from llama_stack.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter


@pytest.fixture
def vertex_config() -> VertexAIConfig:
    return VertexAIConfig(project="test-project", location="global")


@pytest.fixture
def adapter(vertex_config: VertexAIConfig) -> VertexAIInferenceAdapter:
    return VertexAIInferenceAdapter(config=vertex_config)


@pytest.fixture(autouse=True)
def clear_client_cache():
    VertexAIInferenceAdapter._create_adc_client.cache_clear()
    yield
    VertexAIInferenceAdapter._create_adc_client.cache_clear()


class TestVertexAIAdapterInit:
    def test_init_sets_config_and_default_client(
        self, adapter: VertexAIInferenceAdapter, vertex_config: VertexAIConfig
    ):
        assert adapter.config == vertex_config
        assert adapter._default_client is None

    async def test_initialize_sets_default_client(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        client = object()

        monkeypatch.setattr(adapter, "_create_client", lambda **kwargs: client)

        await adapter.initialize()

        assert adapter._default_client is client

    async def test_initialize_failure_keeps_default_client_unset(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        def _raise(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(adapter, "_create_client", _raise)

        await adapter.initialize()

        assert adapter._default_client is None


class TestVertexAIClientManagement:
    def test_create_client_with_access_token_uses_credentials(self, monkeypatch):
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("llama_stack.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        client = VertexAIInferenceAdapter._create_client(
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
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        default_client = object()
        adapter._default_client = cast(Any, default_client)
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: None)

        assert adapter._get_client() is default_client

    def test_get_client_uses_provider_override_with_token(self, monkeypatch):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        override = VertexAIProviderDataValidator(
            vertex_project="override-project",
            vertex_location="us-central1",
            vertex_access_token="override-token",
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
        """Overriding only project/location should still use the configured access_token."""
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
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: None)

        with pytest.raises(RuntimeError, match="No VertexAI client available"):
            adapter._get_client()


class TestVertexAIModelListing:
    async def test_list_provider_model_ids_filters_and_deduplicates(self, monkeypatch):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        models = [
            SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-2.5-pro", supported_actions=[]),
            SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
            SimpleNamespace(name="", supported_actions=["generateContent"]),
        ]

        async def fake_list(**kwargs):
            return _async_pager(models)

        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(list=fake_list)))
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        result = await adapter.list_provider_model_ids()

        assert result == ["google/gemini-2.5-flash", "google/gemini-2.5-pro"]

    async def test_list_models_returns_model_objects(self, monkeypatch):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        adapter.__provider_id__ = "vertexai"
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["google/gemini-2.5-flash", "google/gemini-2.5-pro"]),
        )

        models = await adapter.list_models()

        assert models is not None
        assert len(models) == 2
        assert models[0].identifier == "google/gemini-2.5-flash"
        assert models[0].provider_resource_id == "google/gemini-2.5-flash"
        assert models[0].provider_id == "vertexai"
        assert models[1].identifier == "google/gemini-2.5-pro"

    @pytest.mark.parametrize(
        "allowed",
        [
            ["google/gemini-2.5-flash"],
            ["gemini-2.5-flash"],
        ],
    )
    async def test_list_models_respects_allowed_models(self, monkeypatch, allowed):
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=allowed),
        )
        adapter.__provider_id__ = "vertexai"
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["google/gemini-2.5-flash", "google/gemini-2.5-pro"]),
        )

        models = await adapter.list_models()

        assert models is not None
        assert len(models) == 1
        assert models[0].identifier == "google/gemini-2.5-flash"

    async def test_list_models_propagates_errors(self, monkeypatch):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(side_effect=RuntimeError("API unreachable")),
        )

        with pytest.raises(RuntimeError, match="API unreachable"):
            await adapter.list_models()

    async def test_should_refresh_models_returns_config_value(self):
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
            ("google/gemini-2.5-flash", ["google/gemini-2.5-flash"], None, True),
            ("gemini-2.5-flash", ["google/gemini-2.5-flash"], None, True),
            ("google/gemini-2.5-flash", ["gemini-2.5-flash"], None, True),
            ("google/nonexistent-model", ["google/gemini-2.5-flash"], None, False),
            ("anything", None, RuntimeError("offline"), True),
        ],
    )
    async def test_check_model_availability(self, monkeypatch, model, available_models, error, expected):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        if error is not None:
            monkeypatch.setattr(adapter, "list_provider_model_ids", AsyncMock(side_effect=error))
        else:
            monkeypatch.setattr(adapter, "list_provider_model_ids", AsyncMock(return_value=available_models))

        assert await adapter.check_model_availability(model) is expected


class TestVertexAIAllowedModelsValidation:
    @pytest.mark.parametrize(
        "allowed,requested",
        [
            (["google/gemini-2.5-flash"], "google/gemini-2.5-flash"),
            (["google/gemini-2.5-flash"], "gemini-2.5-flash"),
            (["gemini-2.5-flash"], "google/gemini-2.5-flash"),
            (["gemini-2.5-flash"], "gemini-2.5-flash"),
        ],
    )
    def test_validate_allowed_model_passes_with_or_without_prefix(self, allowed, requested):
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=allowed),
        )
        adapter._validate_model_allowed(requested)

    def test_validate_disallowed_model_raises(self):
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=["google/gemini-2.5-flash"]),
        )
        with pytest.raises(ValueError, match="not in the allowed models list"):
            adapter._validate_model_allowed("gemini-2.5-pro")

    def test_validate_no_allowed_models_passes_anything(self):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        adapter._validate_model_allowed("any-model-at-all")


class TestVertexAIUnsupportedOps:
    @pytest.mark.parametrize(
        "method_name,call_kwargs,error_pattern",
        [
            ("openai_completion", {"params": cast(Any, None)}, "does not support text completions"),
            ("openai_embeddings", {"params": cast(Any, None)}, "embeddings not yet implemented"),
            ("rerank", {"request": cast(Any, None)}, "rerank not yet implemented"),
        ],
    )
    async def test_unsupported_operations_raise(self, method_name, call_kwargs, error_pattern):
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        with pytest.raises(NotImplementedError, match=error_pattern):
            await getattr(adapter, method_name)(**call_kwargs)
