# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from typing_extensions import runtime_checkable

from llama_stack_api.openai_responses import MCPListToolsTool
from llama_stack_api.registries import ListRegistriesResponse, Registry
from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type, webmethod
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA


@json_schema_type
class ConnectorType(StrEnum):
    """Type of connector."""

    MCP = "mcp"


@json_schema_type
class Connector(Resource):
    """A connector resource representing a connector registered in Llama Stack.

    :param type: Type of resource, always 'connector' for connectors
    :param connector_type: Type of connector (e.g., MCP)
    :param connector_id: User-specified identifier for the connector
    :param url: URL of the connector
    :param created_at: Timestamp of creation
    :param updated_at: Timestamp of last update
    :param server_label: (Optional) Label of the server
    :param server_description: (Optional) Description of the server
    :param tools: (Optional) List of tools available from the connector
    :param registry_id: (Optional) ID of the registry this connector belongs to
    """

    model_config = {"populate_by_name": True}

    type: Literal[ResourceType.connector] = ResourceType.connector
    connector_type: ConnectorType = Field(default=ConnectorType.MCP)
    user_connector_id: str | None = Field(
        default=None, alias="connector_id", description="User-specified identifier for the connector"
    )
    url: str = Field(..., description="URL of the connector")
    created_at: datetime = Field(..., description="Timestamp of creation")
    updated_at: datetime = Field(..., description="Timestamp of last update")
    server_name: str | None = Field(default=None, description="Name of the server")
    server_label: str | None = Field(default=None, description="Label of the server")
    server_description: str | None = Field(default=None, description="Description of the server")
    tools: list[MCPListToolsTool] | None = Field(default=None, description="List of tools available from the connector")
    registry_id: str | None = Field(default=None, description="ID of the registry this connector belongs to")

    def _generate_connector_id(self) -> str:
        name = self.server_name if self.server_name is not None else self.identifier
        if self.registry_id is not None:
            return f"{self.connector_type.value}::{self.registry_id}::{name}"
        return f"{self.connector_type.value}::{name}"

    @property
    def connector_id(self) -> str:
        return self.user_connector_id if self.user_connector_id is not None else self._generate_connector_id()


@json_schema_type
class ConnectorInput(BaseModel):
    """Input for creating a connector.

    :param connector_type: Type of connector
    :param connector_id: Unique identifier for the connector
    :param url: URL of the connector
    """

    connector_type: ConnectorType = Field(default=ConnectorType.MCP)
    connector_id: str | None = Field(default=None, description="Unique identifier for the connector")
    url: str = Field(..., description="URL of the connector")


@json_schema_type
class ListConnectorsResponse(BaseModel):
    """Response containing a list of connectors.

    :param data: List of connectors
    """

    data: list[Connector]


@json_schema_type
class ListToolsResponse(BaseModel):
    """Response containing a list of tools.

    :param data: List of tools
    """

    data: list[MCPListToolsTool]


@runtime_checkable
class Connectors(Protocol):
    @webmethod(route="/connectors", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_connectors(
        self,
        registry_id: str | None = None,
        include_tools: bool = False,
    ) -> ListConnectorsResponse:
        """List all configured connectors.

        :param registry_id: (Optional) The ID of a registry to filter connectors for.
        :param include_tools: (Optional) Whether to include tools in the response.
        :returns: A ListConnectorsResponse.
        """
        ...

    @webmethod(route="/connectors/{connector_id:path}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_connector(
        self,
        connector_id: str,
    ) -> Connector:
        """Get a connector by its ID.

        :param connector_id: The ID of the connector to get.
        :returns: A Connector.
        """
        ...

    @webmethod(route="/connectors/{connector_id:path}/tools", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_connector_tools(
        self,
        connector_id: str,
    ) -> ListToolsResponse:
        """List tools available from a connector.

        :param connector_id: The ID of the connector to list tools for.
        :returns: A ListToolsResponse.
        """
        ...

    @webmethod(
        route="/connectors/{connector_id:path}/tools/{tool_name:path}", method="GET", level=LLAMA_STACK_API_V1ALPHA
    )
    async def get_connector_tool(
        self,
        connector_id: str,
        tool_name: str,
    ) -> MCPListToolsTool:
        """Get a tool definition by its name from a connector.

        :param connector_id: The ID of the connector to get the tool from.
        :param tool_name: The name of the tool to get.
        :returns: A MCPListToolsTool.
        """
        ...

    @webmethod(route="/connectors/registries", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_registries(self) -> ListRegistriesResponse:
        """List all registries.

        :returns: A ListRegistriesResponse.
        """
        ...

    @webmethod(route="/connectors/registries/{registry_id:path}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_registry(self, registry_id: str) -> Registry:
        """Get a registry by its ID.

        :param registry_id: The ID of the registry to get.
        :returns: A Registry.
        """
        ...
