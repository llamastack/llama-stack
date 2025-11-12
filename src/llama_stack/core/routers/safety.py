# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.inference import OpenAIMessageParam
from llama_stack.apis.safety import RunShieldResponse, Safety
from llama_stack.apis.safety.safety import ModerationObject
from llama_stack.apis.shields import Shield
from llama_stack.core.datatypes import SafetyConfig
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core::routers")


class SafetyRouter(Safety):
    def __init__(
        self,
        routing_table: RoutingTable,
        safety_config: SafetyConfig | None = None,
    ) -> None:
        logger.debug("Initializing SafetyRouter")
        self.routing_table = routing_table
        self.safety_config = safety_config

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
        messages: list[OpenAIMessageParam],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        logger.debug(f"SafetyRouter.run_shield: {shield_id}")
        provider = await self.routing_table.get_provider_impl(shield_id)
        return await provider.run_shield(
            shield_id=shield_id,
            messages=messages,
            params=params,
        )

    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        list_shields_response = await self.routing_table.list_shields()
        shields = list_shields_response.data

        selected_shield: Shield | None = None
        provider_model: str | None = model

        if model:
            matches: list[Shield] = [s for s in shields if model == s.provider_resource_id]
            if not matches:
                raise ValueError(
                    f"No shield associated with provider_resource id {model}: choose from {[s.provider_resource_id for s in shields]}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple shields associated with provider_resource id {model}: matched shields {[s.identifier for s in matches]}"
                )
            selected_shield = matches[0]
        else:
            default_shield_id = self.safety_config.default_shield_id if self.safety_config else None
            if not default_shield_id:
                raise ValueError(
                    "No moderation model specified and no default_shield_id configured in safety config: select model "
                    f"from {[s.provider_resource_id or s.identifier for s in shields]}"
                )

            selected_shield = next((s for s in shields if s.identifier == default_shield_id), None)
            if selected_shield is None:
                raise ValueError(
                    f"Default moderation model not found. Choose from {[s.provider_resource_id or s.identifier for s in shields]}."
                )

            provider_model = selected_shield.provider_resource_id

        shield_id = selected_shield.identifier
        logger.debug(f"SafetyRouter.run_moderation: {shield_id}")
        provider = await self.routing_table.get_provider_impl(shield_id)

        response = await provider.run_moderation(
            input=input,
            model=provider_model,
        )

        return response
