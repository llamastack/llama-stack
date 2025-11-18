# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test suite for the Dana agent provider implementation (stub).

TODO: Add tests when implementation is complete.
"""

from llama_stack.core.datatypes import Api
from llama_stack.core.distribution import get_provider_registry
from llama_stack.providers.inline.agents.dana.agents import DanaAgentsImpl
from llama_stack.providers.inline.agents.dana.config import DanaAgentsImplConfig


def test_dana_provider_in_registry():
    """Test that the Dana provider is registered and can be found in the registry."""
    provider_registry = get_provider_registry()
    agents_providers = provider_registry.get(Api.agents, {})

    # Verify the provider is in the registry
    assert "inline::dana" in agents_providers, "Dana provider should be registered"

    provider_spec = agents_providers["inline::dana"]
    assert provider_spec.provider_type == "inline::dana"
    assert provider_spec.api == Api.agents
    assert provider_spec.module == "llama_stack.providers.inline.agents.dana"
    assert provider_spec.config_class == "llama_stack.providers.inline.agents.dana.DanaAgentsImplConfig"


def test_dana_provider_config():
    """Test that the Dana provider config can be instantiated."""
    config = DanaAgentsImplConfig.sample_run_config(__distro_dir__="test")
    assert isinstance(config, dict)
    assert "persistence" in config
    assert "agent_state" in config["persistence"]
    assert "responses" in config["persistence"]


def test_dana_provider_class_exists():
    """Test that Dana provider class exists."""
    assert DanaAgentsImpl is not None
    # TODO: Add actual tests when the provider is implemented
