# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Core prompts service delegating to inline::reference provider.

This module provides backward compatibility by delegating to the
inline::reference provider implementation.
"""

from llama_stack.providers.inline.prompts.reference import (
    PromptServiceImpl,
    ReferencePromptsConfig,
    get_adapter_impl,
)

# Re-export for backward compatibility
PromptServiceConfig = ReferencePromptsConfig
get_provider_impl = get_adapter_impl

__all__ = [
    "PromptServiceImpl",
    "PromptServiceConfig",
    "ReferencePromptsConfig",
    "get_provider_impl",
    "get_adapter_impl",
]
