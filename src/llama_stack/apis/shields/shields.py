# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.common.tracing import telemetry_traceable
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, webmethod


class CommonShieldFields(BaseModel):
    params: dict[str, Any] | None = None


@json_schema_type
class Shield(CommonShieldFields, Resource):
    """A safety shield resource that can be used to check content.

    :param params: (Optional) Configuration parameters for the shield
    :param type: The resource type, always shield
    """

    type: Literal[ResourceType.shield] = ResourceType.shield

    @property
    def shield_id(self) -> str:
        return self.identifier

    @property
    def provider_shield_id(self) -> str | None:
        return self.provider_resource_id


class ShieldInput(CommonShieldFields):
    shield_id: str
    provider_id: str | None = None
    provider_shield_id: str | None = None


class ListShieldsResponse(BaseModel):
    data: list[Shield]


@runtime_checkable
@telemetry_traceable
class Shields(Protocol):
    @webmethod(route="/shields", method="GET", level=LLAMA_STACK_API_V1)
    async def list_shields(self) -> ListShieldsResponse:
        """List all shields.

        :returns: A ListShieldsResponse.
        """
        ...

    @webmethod(route="/shields/{identifier:path}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_shield(self, identifier: str) -> Shield:
        """Get a shield by its identifier.

        :param identifier: The identifier of the shield to get.
        :returns: A Shield.
        """
        ...

    @webmethod(route="/shields", method="POST", level=LLAMA_STACK_API_V1, deprecated=True)
    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: str | None = None,
        provider_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Shield:
        """Register a shield.

        :param shield_id: The identifier of the shield to register.
        :param provider_shield_id: The identifier of the shield in the provider.
        :param provider_id: The identifier of the provider.
        :param params: The parameters of the shield.
        :returns: A Shield.
        """
        ...

    @webmethod(route="/shields/{identifier:path}", method="DELETE", level=LLAMA_STACK_API_V1, deprecated=True)
    async def unregister_shield(self, identifier: str) -> None:
        """Unregister a shield.

        :param identifier: The identifier of the shield to unregister.
        """
        ...
