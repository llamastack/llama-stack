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
        config=config,
        inference_api=deps[Api.inference],
        vector_io_api=deps[Api.vector_io],
        safety_api=deps[Api.safety],
        tool_runtime_api=deps[Api.tool_runtime],
        tool_groups_api=deps[Api.tool_groups],
        conversations_api=deps[Api.conversations],
        prompts_api=deps[Api.prompts],
        files_api=deps[Api.files],
        telemetry_enabled=Api.telemetry in deps,
        policy=policy,
    )
    await impl.initialize()
    return impl
