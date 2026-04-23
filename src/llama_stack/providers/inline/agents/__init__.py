# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import AgentsConfig  # noqa: F401
from .impl import BuiltinAgentsImpl  # noqa: F401

__all__ = ["AgentsConfig", "BuiltinAgentsImpl"]


async def get_provider_impl(config: AgentsConfig, _deps):
    impl = BuiltinAgentsImpl(config)
    await impl.initialize()
    return impl
