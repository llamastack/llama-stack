# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type


class CommonShieldFields(BaseModel):
    params: dict[str, Any] | None = Field(default=None, description="Configuration parameters for the shield")


@json_schema_type
class Shield(CommonShieldFields, Resource):
    """A safety shield resource that can be used to check content."""

    type: Literal[ResourceType.shield] = Field(
        default=ResourceType.shield, description="The resource type, always shield"
    )

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
    """Response model for listing shields."""

    data: list[Shield] = Field(..., description="List of shield resources")


@json_schema_type
class RegisterShieldRequest(BaseModel):
    """Request model for registering a shield."""

    shield_id: str = Field(..., description="The identifier of the shield to register")
    provider_shield_id: str | None = Field(default=None, description="The identifier of the shield in the provider")
    provider_id: str | None = Field(default=None, description="The identifier of the provider")
    params: dict[str, Any] | None = Field(default=None, description="The parameters of the shield")
