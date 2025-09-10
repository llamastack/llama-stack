# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import BaseModel

from llama_stack.core.distribution import get_provider_registry, providable_apis
from llama_stack.core.utils.dynamic import instantiate_class_type


class TestProviderConfigurations:
    """Test suite for testing provider configurations across all API types."""

    @pytest.mark.parametrize("api", providable_apis())
    def test_api_providers(self, api):
        provider_registry = get_provider_registry()
        providers = provider_registry.get(api, {})

        failures = []
        for provider_type, provider_spec in providers.items():
            try:
                self._verify_provider_config(provider_type, provider_spec)
            except Exception as e:
                failures.append(f"Failed to verify {provider_type} config: {str(e)}")

        if failures:
            pytest.fail("\n".join(failures))

    def _verify_provider_config(self, provider_type, provider_spec):
        """Helper method to verify a single provider configuration."""
        # Get the config class
        config_class_name = provider_spec.config_class
        config_type = instantiate_class_type(config_class_name)

        assert issubclass(config_type, BaseModel), f"{config_class_name} is not a subclass of BaseModel"

        assert hasattr(config_type, "sample_run_config"), f"{config_class_name} does not have sample_run_config method"

        sample_config = config_type.sample_run_config(__distro_dir__="foobarbaz")
        assert isinstance(sample_config, dict), f"{config_class_name}.sample_run_config() did not return a dict"
