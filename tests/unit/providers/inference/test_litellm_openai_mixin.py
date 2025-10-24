# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin


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
