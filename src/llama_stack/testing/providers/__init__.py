# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Provider-specific exception handling for test recording/replay.

This module enables accurate exception replay during integration tests. When tests
run in "record" mode, exceptions from provider SDKs are serialized. In "replay"
mode, they are reconstructed to match the original exception type.

WHEN TO ADD A NEW PROVIDER:
Add a new provider module when:
1. You have a provider that raises SDK-specific exceptions (with status_code attribute)
2. These exceptions need to be caught and handled differently than generic exceptions
3. The provider's exception module prefix differs from existing providers
4. The provider raises exceptions that do not get translated to OpenAI exceptions by OpenAIMixin

HOW TO ADD A NEW PROVIDER:
1. Create a new file: providers/<provider_name>.py
2. Define these required exports:
   - NAME: str - The provider name (e.g., "anthropic")
   - MODULE_PREFIX: str - The exception's module prefix (e.g., "anthropic" for anthropic.*)
   - create_error(status_code: int, body: dict | None, message: str) -> Exception
3. Import and add to _PROVIDER_MODULES list below

Example provider module (providers/example.py):
    from example_sdk import APIError

    def _create_error(status_code: int, body: dict | None, message: str) -> Exception:
        return APIError(message=message, status_code=status_code)

    NAME = "example"
    MODULE_PREFIX = "example_sdk"
    create_error = _create_error
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from . import ollama, openai

# List of provider modules - add new providers here
_PROVIDER_MODULES: list[ModuleType] = [openai, ollama]


@dataclass
class ProviderConfig:
    """Configuration for a provider's exception handling."""

    module_prefix: str
    create_error: Callable[[int, dict | None, str], Exception]


# Build registry from provider modules
PROVIDERS: dict[str, ProviderConfig] = {
    module.NAME: ProviderConfig(
        module_prefix=module.MODULE_PREFIX,
        create_error=module.create_error,
    )
    for module in _PROVIDER_MODULES
}


def detect_provider(exc: object) -> str:
    """Detect the provider from an exception's module."""
    module = type(exc).__module__
    for name, config in PROVIDERS.items():
        if module.startswith(config.module_prefix):
            return name
    return "unknown"


def create_provider_error(provider: str, status_code: int, body: dict | None, message: str) -> Exception:
    """Reconstruct a provider-specific error from recorded data."""
    if provider in PROVIDERS:
        return PROVIDERS[provider].create_error(status_code, body, message)

    # Fallback for unknown providers
    from llama_stack.testing.exception_utils import GenericProviderError

    return GenericProviderError(status_code, body, message)
