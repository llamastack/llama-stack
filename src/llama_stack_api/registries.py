# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class RegistryType(StrEnum):
    """Type of registry."""

    MCP = "mcp"


@json_schema_type
class Registry(Resource):
    """A registry resource representing a registry of connectors.

    :param type: Type of resource, always 'registry' for registries
    :param identifier: Unique identifier for this resource in llama stack
    :param provider_resource_id: Unique identifier for this resource in the provider
    :param provider_id: ID of the provider that owns this resource
    :param type: Type of resource (e.g. 'model', 'shield', 'vector_store', etc.)
    """

    type: Literal[ResourceType.registry] = ResourceType.registry
    registry_type: RegistryType = Field(default=RegistryType.MCP)
    user_registry_id: str | None = Field(default=None, description="User-specified identifier for the registry")
    url: str = Field(..., description="URL of the registry")
    created_at: datetime = Field(..., description="Timestamp of creation")
    updated_at: datetime = Field(..., description="Timestamp of last update")

    @property
    def registry_id(self) -> str:
        return self.user_registry_id if self.user_registry_id is not None else self.identifier


@json_schema_type
class ListRegistriesResponse(BaseModel):
    """Response containing a list of registries.

    :param data: List of registries
    """

    data: list[Registry]


@json_schema_type
class RegistryInput(BaseModel):
    """Input for creating a registry.

    :param url: URL of the registry
    :param user_registry_id: User-specified identifier for the registry
    """

    url: str
    user_registry_id: str | None = Field(default=None, description="User-specified identifier for the registry")
    registry_type: RegistryType = Field(default=RegistryType.MCP)
