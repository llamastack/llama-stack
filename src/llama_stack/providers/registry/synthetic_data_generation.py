# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return available providers for synthetic data generation API."""
    return []
