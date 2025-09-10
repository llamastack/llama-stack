# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.resource import ResourceType
from llama_stack.apis.shields import ListShieldsResponse, Shield, Shields
from llama_stack.core.datatypes import (
    ShieldWithOwner,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    async def list_shields(self) -> ListShieldsResponse:
        return ListShieldsResponse(data=await self.get_all_with_type(ResourceType.shield.value))

    async def get_shield(self, identifier: str) -> Shield:
        shield = await self.get_object_by_identifier("shield", identifier)
        if shield is None:
            raise ValueError(f"Shield '{identifier}' not found")
        return shield

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: str | None = None,
        provider_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Shield:
        if provider_shield_id is None:
            provider_shield_id = shield_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this shield type
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        if params is None:
            params = {}
        shield = ShieldWithOwner(
            identifier=shield_id,
            provider_resource_id=provider_shield_id,
            provider_id=provider_id,
            params=params,
        )
        await self.register_object(shield)
        return shield

    async def unregister_shield(self, identifier: str) -> None:
        existing_shield = await self.get_shield(identifier)
        await self.unregister_object(existing_shield)
