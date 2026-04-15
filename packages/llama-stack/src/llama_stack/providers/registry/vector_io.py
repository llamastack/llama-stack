# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import (
    Api,
    ProviderSpec,
)

# Common dependencies for all vector IO providers that support document processing
DEFAULT_VECTOR_IO_DEPS = ["chardet", "pypdf>=6.10.0"]


def available_providers() -> list[ProviderSpec]:
    """Return the list of available vector I/O provider specifications.

    All vector I/O providers are now discovered via entry points.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    from llama_stack.providers.registry import merge_entry_point_providers

    return merge_entry_point_providers([], api=Api.vector_io)
