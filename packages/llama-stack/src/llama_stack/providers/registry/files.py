# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return the list of available files provider specifications.

    All files providers are discovered via entry points.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    from llama_stack.providers.registry import merge_entry_point_providers

    return merge_entry_point_providers([], api=Api.files)
