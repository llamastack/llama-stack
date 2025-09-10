# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from unittest.mock import patch

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError

from llama_stack.core.datatypes import Api, Provider, StackRunConfig
from llama_stack.core.distribution import get_provider_registry
from llama_stack.providers.datatypes import ProviderSpec


class SampleConfig(BaseModel):
    foo: str = Field(
        default="bar",
        description="foo",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "foo": "baz",
        }


@pytest.fixture
def mock_providers():
    """Mock the available_providers function to return test providers."""
    with patch("llama_stack.providers.registry.inference.available_providers") as mock:
        mock.return_value = [
            ProviderSpec(
                provider_type="test_provider",
                api=Api.inference,
                adapter_type="test_adapter",
                config_class="test_provider.config.TestProviderConfig",
            )
        ]
        yield mock


@pytest.fixture
def base_config(tmp_path):
    """Create a base StackRunConfig with common settings."""
    return StackRunConfig(
        image_name="test_image",
        providers={
            "inference": [
                Provider(
                    provider_id="sample_provider",
                    provider_type="sample",
                    config=SampleConfig.sample_run_config(),
                )
            ]
        },
        external_providers_dir=str(tmp_path),
    )


@pytest.fixture
def provider_spec_yaml():
    """Common provider spec YAML for testing."""
    return """
adapter:
  adapter_type: test_provider
  config_class: test_provider.config.TestProviderConfig
  module: test_provider
api_dependencies:
  - safety
"""


@pytest.fixture
def inline_provider_spec_yaml():
    """Common inline provider spec YAML for testing."""
    return """
module: test_provider
config_class: test_provider.config.TestProviderConfig
pip_packages:
  - test-package
api_dependencies:
  - safety
optional_api_dependencies:
  - vector_io
provider_data_validator: test_provider.validator.TestValidator
container_image: test-image:latest
"""


@pytest.fixture
def api_directories(tmp_path):
    """Create the API directory structure for testing."""
    # Create remote provider directory
    remote_inference_dir = tmp_path / "remote" / "inference"
    remote_inference_dir.mkdir(parents=True, exist_ok=True)

    # Create inline provider directory
    inline_inference_dir = tmp_path / "inline" / "inference"
    inline_inference_dir.mkdir(parents=True, exist_ok=True)

    return remote_inference_dir, inline_inference_dir


def make_import_module_side_effect(
    builtin_provider_spec=None,
    external_module=None,
    raise_for_external=False,
    missing_get_provider_spec=False,
):
    from types import SimpleNamespace

    def import_module_side_effect(name):
        if name == "llama_stack.providers.registry.inference":
            mock_builtin = SimpleNamespace(
                available_providers=lambda: [
                    builtin_provider_spec
                    or ProviderSpec(
                        api=Api.inference,
                        provider_type="test_provider",
                        config_class="test_provider.config.TestProviderConfig",
                        module="test_provider",
                    )
                ]
            )
            return mock_builtin
        elif name == "external_test.provider":
            if raise_for_external:
                raise ModuleNotFoundError(name)
            if missing_get_provider_spec:
                return SimpleNamespace()
            return external_module
        else:
            raise ModuleNotFoundError(name)

    return import_module_side_effect


class TestProviderRegistry:
    """Test suite for provider registry functionality."""

    def test_builtin_providers(self, mock_providers):
        """Test loading built-in providers."""
        registry = get_provider_registry(None)

        assert Api.inference in registry
        assert "test_provider" in registry[Api.inference]
        assert registry[Api.inference]["test_provider"].provider_type == "test_provider"
        assert registry[Api.inference]["test_provider"].api == Api.inference

    def test_external_remote_providers(self, api_directories, mock_providers, base_config, provider_spec_yaml):
        """Test loading external remote providers from YAML files."""
        remote_dir, _ = api_directories
        with open(remote_dir / "test_provider.yaml", "w") as f:
            f.write(provider_spec_yaml)

        registry = get_provider_registry(base_config)
        assert len(registry[Api.inference]) == 2

        assert Api.inference in registry
        assert "remote::test_provider" in registry[Api.inference]
        provider = registry[Api.inference]["remote::test_provider"]
        assert provider.adapter.adapter_type == "test_provider"
        assert provider.adapter.module == "test_provider"
        assert provider.adapter.config_class == "test_provider.config.TestProviderConfig"
        assert Api.safety in provider.api_dependencies

    def test_external_inline_providers(self, api_directories, mock_providers, base_config, inline_provider_spec_yaml):
        """Test loading external inline providers from YAML files."""
        _, inline_dir = api_directories
        with open(inline_dir / "test_provider.yaml", "w") as f:
            f.write(inline_provider_spec_yaml)

        registry = get_provider_registry(base_config)
        assert len(registry[Api.inference]) == 2

        assert Api.inference in registry
        assert "inline::test_provider" in registry[Api.inference]
        provider = registry[Api.inference]["inline::test_provider"]
        assert provider.provider_type == "inline::test_provider"
        assert provider.module == "test_provider"
        assert provider.config_class == "test_provider.config.TestProviderConfig"
        assert provider.pip_packages == ["test-package"]
        assert Api.safety in provider.api_dependencies
        assert Api.vector_io in provider.optional_api_dependencies
        assert provider.provider_data_validator == "test_provider.validator.TestValidator"
        assert provider.container_image == "test-image:latest"

    def test_invalid_yaml(self, api_directories, mock_providers, base_config):
        """Test handling of invalid YAML files."""
        remote_dir, inline_dir = api_directories
        with open(remote_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: -")
        with open(inline_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: -")

        with pytest.raises(yaml.YAMLError):
            get_provider_registry(base_config)

    def test_missing_directory(self, mock_providers):
        """Test handling of missing external providers directory."""
        config = StackRunConfig(
            image_name="test_image",
            providers={
                "inference": [
                    Provider(
                        provider_id="sample_provider",
                        provider_type="sample",
                        config=SampleConfig.sample_run_config(),
                    )
                ]
            },
            external_providers_dir="/nonexistent/dir",
        )
        with pytest.raises(FileNotFoundError):
            get_provider_registry(config)

    def test_empty_api_directory(self, api_directories, mock_providers, base_config):
        """Test handling of empty API directory."""
        registry = get_provider_registry(base_config)
        assert len(registry[Api.inference]) == 1  # Only built-in provider

    def test_malformed_remote_provider_spec(self, api_directories, mock_providers, base_config):
        """Test handling of malformed remote provider spec (missing required fields)."""
        remote_dir, _ = api_directories
        malformed_spec = """
adapter:
  adapter_type: test_provider
  # Missing required fields
api_dependencies:
  - safety
"""
        with open(remote_dir / "malformed.yaml", "w") as f:
            f.write(malformed_spec)

        with pytest.raises(ValidationError):
            get_provider_registry(base_config)

    def test_malformed_inline_provider_spec(self, api_directories, mock_providers, base_config):
        """Test handling of malformed inline provider spec (missing required fields)."""
        _, inline_dir = api_directories
        malformed_spec = """
module: test_provider
# Missing required config_class
pip_packages:
  - test-package
"""
        with open(inline_dir / "malformed.yaml", "w") as f:
            f.write(malformed_spec)

        with pytest.raises(KeyError) as exc_info:
            get_provider_registry(base_config)
        assert "config_class" in str(exc_info.value)

    def test_external_provider_from_module_success(self, mock_providers):
        """Test loading an external provider from a module (success path)."""
        from types import SimpleNamespace

        from llama_stack.core.datatypes import Provider, StackRunConfig
        from llama_stack.providers.datatypes import Api, ProviderSpec

        # Simulate a provider module with get_provider_spec
        fake_spec = ProviderSpec(
            api=Api.inference,
            provider_type="external_test",
            config_class="external_test.config.ExternalTestConfig",
            module="external_test",
        )
        fake_module = SimpleNamespace(get_provider_spec=lambda: fake_spec)

        import_module_side_effect = make_import_module_side_effect(external_module=fake_module)

        with patch("importlib.import_module", side_effect=import_module_side_effect) as mock_import:
            config = StackRunConfig(
                image_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="external_test",
                            provider_type="external_test",
                            config={},
                            module="external_test",
                        )
                    ]
                },
            )
            registry = get_provider_registry(config)
            assert Api.inference in registry
            assert "external_test" in registry[Api.inference]
            provider = registry[Api.inference]["external_test"]
            assert provider.module == "external_test"
            assert provider.config_class == "external_test.config.ExternalTestConfig"
            mock_import.assert_any_call("llama_stack.providers.registry.inference")
            mock_import.assert_any_call("external_test.provider")

    def test_external_provider_from_module_not_found(self, mock_providers):
        """Test handling ModuleNotFoundError for missing provider module."""
        from llama_stack.core.datatypes import Provider, StackRunConfig

        import_module_side_effect = make_import_module_side_effect(raise_for_external=True)

        with patch("importlib.import_module", side_effect=import_module_side_effect):
            config = StackRunConfig(
                image_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="external_test",
                            provider_type="external_test",
                            config={},
                            module="external_test",
                        )
                    ]
                },
            )
            with pytest.raises(ValueError) as exc_info:
                get_provider_registry(config)
            assert "get_provider_spec not found" in str(exc_info.value)

    def test_external_provider_from_module_missing_get_provider_spec(self, mock_providers):
        """Test handling missing get_provider_spec in provider module (should raise ValueError)."""
        from llama_stack.core.datatypes import Provider, StackRunConfig

        import_module_side_effect = make_import_module_side_effect(missing_get_provider_spec=True)

        with patch("importlib.import_module", side_effect=import_module_side_effect):
            config = StackRunConfig(
                image_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="external_test",
                            provider_type="external_test",
                            config={},
                            module="external_test",
                        )
                    ]
                },
            )
            with pytest.raises(AttributeError):
                get_provider_registry(config)

    def test_external_provider_from_module_building(self, mock_providers):
        """Test loading an external provider from a module during build (building=True, partial spec)."""
        from llama_stack.core.datatypes import BuildConfig, BuildProvider, DistributionSpec
        from llama_stack.providers.datatypes import Api

        # No importlib patch needed, should not import module when type of `config` is BuildConfig or DistributionSpec
        build_config = BuildConfig(
            version=2,
            image_type="container",
            image_name="test_image",
            distribution_spec=DistributionSpec(
                description="test",
                providers={
                    "inference": [
                        BuildProvider(
                            provider_type="external_test",
                            module="external_test",
                        )
                    ]
                },
            ),
        )
        registry = get_provider_registry(build_config)
        assert Api.inference in registry
        assert "external_test" in registry[Api.inference]
        provider = registry[Api.inference]["external_test"]
        assert provider.module == "external_test"
        assert provider.is_external is True
        # config_class is empty string in partial spec
        assert provider.config_class == ""
