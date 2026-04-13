# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.distribution import discover_entry_point_providers
from llama_stack_api import Api, ProviderSpec


def merge_entry_point_providers(providers: list[ProviderSpec], api: Api) -> list[ProviderSpec]:
    """Merge entry-point-discovered providers into a list, skipping duplicates.

    This allows providers that have been extracted into separate packages
    (and register themselves via entry points) to appear alongside in-tree
    providers.

    Args:
        providers: The existing in-tree provider list.
        api: The API to filter entry-point providers by.

    Returns:
        A new list containing both in-tree and entry-point providers.
    """
    existing_types = {p.provider_type for p in providers}
    merged = list(providers)
    for spec in discover_entry_point_providers(api=api):
        if spec.provider_type not in existing_types:
            merged.append(spec)
            existing_types.add(spec.provider_type)
    return merged
