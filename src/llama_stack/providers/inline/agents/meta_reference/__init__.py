# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import AccessRule, Api

from .config import MetaReferenceAgentsImplConfig


async def get_provider_impl(
    config: MetaReferenceAgentsImplConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
    telemetry_enabled: bool = False,
):
    from .agents import MetaReferenceAgentsImpl

    impl = MetaReferenceAgentsImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps[Api.safety],
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
        deps[Api.conversations],
        policy,
        telemetry_enabled,
    )
    await impl.initialize()
    return impl
