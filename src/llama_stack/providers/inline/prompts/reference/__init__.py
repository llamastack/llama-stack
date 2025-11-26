# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import ReferencePromptsConfig
from .reference import PromptServiceImpl


async def get_adapter_impl(config: ReferencePromptsConfig, _deps):
    impl = PromptServiceImpl(config=config, deps=_deps)
    await impl.initialize()
    return impl


__all__ = ["ReferencePromptsConfig", "PromptServiceImpl", "get_adapter_impl"]
