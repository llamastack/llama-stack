from typing import Any

from llama_stack.core.datatypes import AccessRule, Api

from .config import DanaAgentConfig


async def get_provider_impl(
    config: DanaAgentConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
    telemetry_enabled: bool = False,
):
    from .agents import DanaAgentsImpl

    impl = DanaAgentsImpl(
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


__all__ = ["DanaAgentConfig", "get_provider_impl"]
