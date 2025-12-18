# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from typing_extensions import runtime_checkable

from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type, webmethod
from llama_stack_api.tools import ToolDef
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA


@json_schema_type
class ConnectorType(StrEnum):
    """Type of connector."""

    MCP = "mcp"


class CommonConnectorFields(BaseModel):
    """Common fields for all connectors.

    :param connector_type: Type of connector
    :param connector_id: Identifier for the connector
    :param url: URL of the connector
    :param server_label: (Optional) Label of the server
    """

    connector_type: ConnectorType = Field(default=ConnectorType.MCP)
    connector_id: str = Field(..., description="Identifier for the connector")
    url: str = Field(..., description="URL of the connector")
    server_label: str | None = Field(default=None, description="Label of the server")


@json_schema_type
class Connector(CommonConnectorFields, Resource):
    """A connector resource representing a connector registered in Llama Stack.

    :param type: Type of resource, always 'connector' for connectors
    :param server_name: (Optional) Name of the server
    :param server_description: (Optional) Description of the server
    """

    model_config = {"populate_by_name": True}
    type: Literal[ResourceType.connector] = ResourceType.connector
    server_name: str | None = Field(default=None, description="Name of the server")
    server_description: str | None = Field(default=None, description="Description of the server")


@json_schema_type
class ConnectorInput(CommonConnectorFields):
    """Input for creating a connector

    :param type: Type of resource, always 'connector' for connectors
    """

    type: Literal[ResourceType.connector] = ResourceType.connector


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

    data: list[ToolDef]


@runtime_checkable
class Connectors(Protocol):
    # NOTE: Route order matters! More specific routes must come before less specific ones.
    # Routes with {param:path} are greedy and will match everything including slashes.

    @webmethod(route="/connectors", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_connectors(
        self,
    ) -> ListConnectorsResponse:
        """List all configured connectors.

        :returns: A ListConnectorsResponse.
        """
        ...

    @webmethod(route="/connectors/{connector_id}/tools/{tool_name}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_connector_tool(
        self,
        connector_id: str,
        tool_name: str,
        authorization: str | None = None,
    ) -> ToolDef:
        """Get a tool definition by its name from a connector.

        :param connector_id: The ID of the connector to get the tool from.
        :param tool_name: The name of the tool to get.
        :param authorization: (Optional) OAuth access token for authenticating with the MCP server.

        :returns: A ToolDef.
        """
        ...

    @webmethod(route="/connectors/{connector_id}/tools", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def list_connector_tools(
        self,
        connector_id: str,
        authorization: str | None = None,
    ) -> ListToolsResponse:
        """List tools available from a connector.

        :param connector_id: The ID of the connector to list tools for.
        :param authorization: (Optional) OAuth access token for authenticating with the MCP server.

        :returns: A ListToolsResponse.
        """
        ...

    @webmethod(route="/connectors/{connector_id}", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_connector(
        self,
        connector_id: str,
        authorization: str | None = None,
    ) -> Connector:
        """Get a connector by its ID.

        :param connector_id: The ID of the connector to get.
        :param authorization: (Optional) OAuth access token for authenticating with the MCP server.

        :returns: A Connector.
        """
        ...
