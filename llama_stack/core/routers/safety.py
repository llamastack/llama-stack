# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse, Safety
from llama_stack.apis.safety.safety import ModerationObject
from llama_stack.apis.shields import Shield
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core::routers")


class SafetyRouter(Safety):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing SafetyRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("SafetyRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("SafetyRouter.shutdown")
        pass

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: str | None = None,
        provider_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Shield:
        logger.debug(f"SafetyRouter.register_shield: {shield_id}")
        return await self.routing_table.register_shield(shield_id, provider_shield_id, provider_id, params)

    async def unregister_shield(self, identifier: str) -> None:
        logger.debug(f"SafetyRouter.unregister_shield: {identifier}")
        return await self.routing_table.unregister_shield(identifier)

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        logger.debug(f"SafetyRouter.run_shield: {shield_id}")
        provider = await self.routing_table.get_provider_impl(shield_id)
        return await provider.run_shield(
            shield_id=shield_id,
            messages=messages,
            params=params,
        )

    async def run_moderation(self, input: str | list[str], model: str) -> ModerationObject:
        async def get_shield_id(self, model: str) -> str:
            """Get Shield id from model (provider_resource_id) of shield."""
            list_shields_response = await self.routing_table.list_shields()

            matches = [s.identifier for s in list_shields_response.data if model == s.provider_resource_id]

            if not matches:
                raise ValueError(f"No shield associated with provider_resource id {model}")
            if len(matches) > 1:
                raise ValueError(f"Multiple shields associated with provider_resource id {model}")
            return matches[0]

        shield_id = await get_shield_id(self, model)
        logger.debug(f"SafetyRouter.run_moderation: {shield_id}")
        provider = await self.routing_table.get_provider_impl(shield_id)

        response = await provider.run_moderation(
            input=input,
            model=model,
        )

        return response
