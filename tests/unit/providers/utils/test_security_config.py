# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from pydantic import BaseModel

from llama_stack.providers.utils.common.security_config import (
    DEFAULT_TRUSTED_MODEL_PREFIXES,
    TrustedModelConfig,
)


class MockConfig(TrustedModelConfig, BaseModel):
    """Mock config class for testing TrustedModelConfig mixin"""

    pass


class TestDefaultTrustedModelPrefixes:
    """Test the default trusted model prefixes"""

    def test_default_contains_nomic_ai(self):
        """nomic-ai/ should be in the default list as it requires trust_remote_code"""
        assert "nomic-ai/" in DEFAULT_TRUSTED_MODEL_PREFIXES
        assert len(DEFAULT_TRUSTED_MODEL_PREFIXES) == 1


class TestTrustedModelConfigMixin:
    """Test the TrustedModelConfig mixin functionality"""

    def test_default_initialization(self):
        """Config should initialize with default trusted prefixes"""
        config = MockConfig()
        assert config.trusted_model_prefixes == DEFAULT_TRUSTED_MODEL_PREFIXES

    def test_custom_trusted_prefixes(self):
        """Config should accept custom trusted prefixes"""
        custom_prefixes = ["meta-llama/", "mistralai/"]
        config = MockConfig(trusted_model_prefixes=custom_prefixes)
        assert config.trusted_model_prefixes == custom_prefixes

    def test_empty_trusted_prefixes(self):
        """Config should allow empty trusted prefixes list"""
        config = MockConfig(trusted_model_prefixes=[])
        assert config.trusted_model_prefixes == []


class TestIsTrustedModel:
    """Test the is_trusted_model() method"""

    def test_trusted_model_exact_prefix(self):
        """Model with exact prefix match should be trusted"""
        config = MockConfig(trusted_model_prefixes=["nomic-ai/"])
        assert config.is_trusted_model("nomic-ai/nomic-embed-text-v1.5") is True

    def test_trusted_model_multiple_prefixes(self):
        """Model matching any prefix should be trusted"""
        config = MockConfig(trusted_model_prefixes=["meta-llama/", "mistralai/", "nomic-ai/"])
        assert config.is_trusted_model("meta-llama/Llama-3.3-70B-Instruct") is True
        assert config.is_trusted_model("mistralai/Mistral-7B-v0.1") is True
        assert config.is_trusted_model("nomic-ai/nomic-embed-text-v1.5") is True
        assert config.is_trusted_model("intfloat/multilingual-e5-large") is False

    def test_partial_prefix_no_match(self):
        """Partial prefix match should not be trusted"""
        config = MockConfig(trusted_model_prefixes=["nomic-ai/"])
        assert config.is_trusted_model("nomic-embed-text-v1.5") is False
        assert config.is_trusted_model("nomic") is False

    def test_empty_trusted_list(self):
        """With empty trusted list, no models should be trusted"""
        config = MockConfig(trusted_model_prefixes=[])
        assert config.is_trusted_model("nomic-ai/nomic-embed-text-v1.5") is False
        assert config.is_trusted_model("meta-llama/Llama-3.3-70B") is False

    def test_case_sensitive_matching(self):
        """Prefix matching should be case-sensitive"""
        config = MockConfig(trusted_model_prefixes=["nomic-ai/"])
        assert config.is_trusted_model("nomic-ai/model") is True
        assert config.is_trusted_model("Nomic-AI/model") is False
        assert config.is_trusted_model("NOMIC-AI/model") is False

    def test_model_with_path(self):
        """Models with full paths should work correctly"""
        config = MockConfig(trusted_model_prefixes=["meta-llama/"])
        assert config.is_trusted_model("meta-llama/models/Llama-3.3-70B-Instruct") is True

    def test_model_without_org(self):
        """Models without organization prefix should not be trusted"""
        config = MockConfig(trusted_model_prefixes=["nomic-ai/", "meta-llama/"])
        assert config.is_trusted_model("my-custom-model") is False
        assert config.is_trusted_model("llama-3-8b") is False
