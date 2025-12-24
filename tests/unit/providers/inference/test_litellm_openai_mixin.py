# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack_api import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)


# Test fixtures and helper classes
class FakeConfig(BaseModel):
    api_key: str | None = Field(default=None)


class FakeProviderDataValidator(BaseModel):
    test_api_key: str | None = Field(default=None)


class FakeLiteLLMAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: FakeConfig):
        super().__init__(
            litellm_provider_name="test",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="test_api_key",
            openai_compat_api_base=None,
        )


@pytest.fixture
def adapter_with_config_key():
    """Fixture to create adapter with API key in config"""
    config = FakeConfig(api_key="config-api-key")
    adapter = FakeLiteLLMAdapter(config)
    adapter.__provider_spec__ = MagicMock()
    adapter.__provider_spec__.provider_data_validator = (
        "tests.unit.providers.inference.test_litellm_openai_mixin.FakeProviderDataValidator"
    )
    return adapter


@pytest.fixture
def adapter_without_config_key():
    """Fixture to create adapter without API key in config"""
    config = FakeConfig(api_key=None)
    adapter = FakeLiteLLMAdapter(config)
    adapter.__provider_spec__ = MagicMock()
    adapter.__provider_spec__.provider_data_validator = (
        "tests.unit.providers.inference.test_litellm_openai_mixin.FakeProviderDataValidator"
    )
    return adapter


def test_api_key_from_config_when_no_provider_data(adapter_with_config_key):
    """Test that adapter uses config API key when no provider data is available"""
    api_key = adapter_with_config_key.get_api_key()
    assert api_key == "config-api-key"


def test_provider_data_takes_priority_over_config(adapter_with_config_key):
    """Test that provider data API key overrides config API key"""
    with request_provider_data_context(
        {"x-llamastack-provider-data": json.dumps({"test_api_key": "provider-data-key"})}
    ):
        api_key = adapter_with_config_key.get_api_key()
        assert api_key == "provider-data-key"


def test_fallback_to_config_when_provider_data_missing_key(adapter_with_config_key):
    """Test fallback to config when provider data doesn't have the required key"""
    with request_provider_data_context({"x-llamastack-provider-data": json.dumps({"wrong_key": "some-value"})}):
        api_key = adapter_with_config_key.get_api_key()
        assert api_key == "config-api-key"


def test_error_when_no_api_key_available(adapter_without_config_key):
    """Test that ValueError is raised when neither config nor provider data have API key"""
    with pytest.raises(ValueError, match="API key is not set"):
        adapter_without_config_key.get_api_key()


def test_error_when_provider_data_has_wrong_key(adapter_without_config_key):
    """Test that ValueError is raised when provider data exists but doesn't have required key"""
    with request_provider_data_context({"x-llamastack-provider-data": json.dumps({"wrong_key": "some-value"})}):
        with pytest.raises(ValueError, match="API key is not set"):
            adapter_without_config_key.get_api_key()


def test_provider_data_works_when_config_is_none(adapter_without_config_key):
    """Test that provider data works even when config has no API key"""
    with request_provider_data_context(
        {"x-llamastack-provider-data": json.dumps({"test_api_key": "provider-only-key"})}
    ):
        api_key = adapter_without_config_key.get_api_key()
        assert api_key == "provider-only-key"


def test_error_message_includes_correct_field_names(adapter_without_config_key):
    """Test that error message includes correct field name and header information"""
    try:
        adapter_without_config_key.get_api_key()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "test_api_key" in str(e)  # Should mention the correct field name
        assert "x-llamastack-provider-data" in str(e)  # Should mention header name


class TestLiteLLMOpenAIMixinStreamOptionsInjection:
    """Test cases for automatic stream_options injection in LiteLLMOpenAIMixin"""

    @pytest.fixture
    def mixin_with_model_store(self, adapter_with_config_key):
        """Fixture to create adapter with mocked model store"""
        mock_model_store = AsyncMock()
        mock_model = MagicMock()
        mock_model.provider_resource_id = "test-model-id"
        mock_model_store.get_model = AsyncMock(return_value=mock_model)
        adapter_with_config_key.model_store = mock_model_store
        return adapter_with_config_key

    async def test_chat_completion_injects_stream_options_when_telemetry_active(self, mixin_with_model_store):
        """Test that stream_options is injected for streaming chat completion when telemetry is active"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = MagicMock()

                await mixin_with_model_store.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
                    )
                )

                mock_acompletion.assert_called_once()
                call_kwargs = mock_acompletion.call_args[1]
                assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_chat_completion_preserves_existing_stream_options(self, mixin_with_model_store):
        """Test that existing stream_options are preserved with include_usage added"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = MagicMock()

                await mixin_with_model_store.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=True,
                        stream_options={"other_option": True},
                    )
                )

                call_kwargs = mock_acompletion.call_args[1]
                assert call_kwargs["stream_options"] == {"other_option": True, "include_usage": True}

    async def test_chat_completion_no_injection_when_telemetry_inactive(self, mixin_with_model_store):
        """Test that stream_options is NOT injected when telemetry is inactive"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = MagicMock()

                await mixin_with_model_store.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
                    )
                )

                call_kwargs = mock_acompletion.call_args[1]
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_chat_completion_no_injection_when_not_streaming(self, mixin_with_model_store):
        """Test that stream_options is NOT injected for non-streaming requests"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = MagicMock()

                await mixin_with_model_store.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=False,
                    )
                )

                call_kwargs = mock_acompletion.call_args[1]
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_completion_injects_stream_options_when_telemetry_active(self, mixin_with_model_store):
        """Test that stream_options is injected for streaming completion when telemetry is active"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.atext_completion", new_callable=AsyncMock) as mock_atext_completion:
                mock_atext_completion.return_value = MagicMock()

                await mixin_with_model_store.openai_completion(
                    OpenAICompletionRequestWithExtraBody(model="test-model", prompt="Hello", stream=True)
                )

                mock_atext_completion.assert_called_once()
                call_kwargs = mock_atext_completion.call_args[1]
                assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_completion_no_injection_when_telemetry_inactive(self, mixin_with_model_store):
        """Test that stream_options is NOT injected for completion when telemetry is inactive"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.atext_completion", new_callable=AsyncMock) as mock_atext_completion:
                mock_atext_completion.return_value = MagicMock()

                await mixin_with_model_store.openai_completion(
                    OpenAICompletionRequestWithExtraBody(model="test-model", prompt="Hello", stream=True)
                )

                call_kwargs = mock_atext_completion.call_args[1]
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_original_params_not_mutated(self, mixin_with_model_store):
        """Test that original params object is not mutated when stream_options is injected"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        original_params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
        )

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = MagicMock()

                await mixin_with_model_store.openai_chat_completion(original_params)

                # Original params should not be modified
                assert original_params.stream_options is None

    async def test_chat_completion_overrides_include_usage_false(self, mixin_with_model_store):
        """Test that include_usage=False is overridden when telemetry is active"""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = MagicMock()

                await mixin_with_model_store.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=True,
                        stream_options={"include_usage": False},
                    )
                )

                call_kwargs = mock_acompletion.call_args[1]
                # Telemetry must override False to ensure complete metrics
                assert call_kwargs["stream_options"]["include_usage"] is True
