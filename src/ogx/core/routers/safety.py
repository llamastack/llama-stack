# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.datatypes import SafetyConfig
from ogx.log import get_logger
from ogx_api import (
    ModerationObject,
    RoutingTable,
    RunModerationRequest,
    Safety,
    Shield,
)

logger = get_logger(name=__name__, category="core::routers")


class SafetyRouter(Safety):
    """Router that delegates moderation operations to the appropriate provider via a routing table."""

    def __init__(
        self,
        routing_table: RoutingTable,
        safety_config: SafetyConfig | None = None,
    ) -> None:
        logger.debug("Initializing SafetyRouter")
        self.routing_table = routing_table
        self.safety_config = safety_config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        list_shields_response = await self.routing_table.list_shields()
        shields = list_shields_response.data

        selected_shield: Shield | None = None
        provider_model: str | None = request.model

        if request.model:
            matches: list[Shield] = [s for s in shields if request.model == s.provider_resource_id]
            if not matches:
                raise ValueError(
                    f"No shield associated with provider_resource id {request.model}: choose from {[s.provider_resource_id for s in shields]}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple shields associated with provider_resource id {request.model}: matched shields {[s.identifier for s in matches]}"
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
        logger.debug("SafetyRouter.run_moderation", shield_id=shield_id)
        provider = await self.routing_table.get_provider_impl(shield_id)

        provider_request = RunModerationRequest(input=request.input, model=provider_model)
        return await provider.run_moderation(provider_request)
