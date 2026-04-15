# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.registry.safety import available_providers


def test_passthrough_in_registry():
    providers = available_providers()
    provider_types = [p.provider_type for p in providers]
    assert "remote::passthrough" in provider_types


def test_passthrough_registry_has_provider_data_validator():
    providers = available_providers()
    passthrough = next(p for p in providers if p.provider_type == "remote::passthrough")
    assert passthrough.provider_data_validator is not None
    assert "PassthroughProviderDataValidator" in passthrough.provider_data_validator


def test_passthrough_registry_module_path():
    providers = available_providers()
    passthrough = next(p for p in providers if p.provider_type == "remote::passthrough")
    assert passthrough.module == "llama_stack_provider_safety_passthrough"


def test_passthrough_provider_discovered():
    """Verify the passthrough safety provider is discovered via entry points."""
    providers = available_providers()
    provider_types = [p.provider_type for p in providers]
    assert "remote::passthrough" in provider_types
