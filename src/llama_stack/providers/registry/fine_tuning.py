# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.registry.core import ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Registry of available fine-tuning providers.

    Note: Currently returns an empty list as no providers are implemented yet.
    The fine-tuning API routes are available for OpenAPI spec generation.
    """
    return []
