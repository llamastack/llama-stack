# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.log import get_logger
from llama_stack.providers.utils.tools.mcp import get_mcp_server_info, list_mcp_tools
from llama_stack_api import (
    Connector,
    Connectors,
    ConnectorType,
    ListConnectorsResponse,
    ListRegistriesResponse,
    ListToolsResponse,
    Registry,
    ToolDef,
)
from llama_stack_api.common.errors import (
    ConnectorNotFoundError,
    ConnectorToolNotFoundError,
    RegistryNotFoundError,
)

logger = get_logger(name=__name__, category="connectors")


class ConnectorServiceConfig(BaseModel):
    """Configuration for the built-in connector service.

    :param run_config: Stack run configuration for resolving persistence
    """

    run_config: StackRunConfig


async def get_provider_impl(config: ConnectorServiceConfig):
    """Get the connector service implementation."""
    impl = ConnectorServiceImpl(config)
    return impl


class ConnectorServiceImpl(Connectors):
    """Built-in connector service implementation."""

    def __init__(self, config: ConnectorServiceConfig):
        self.config = config
        # TODO: should these be stored in a kvstore?
        self.connectors_map: dict[str, Connector] = {}
        self.registries_map: dict[str, Registry] = {}

    async def register_connector(
        self,
        url: str,
        connector_id: str | None = None,
        connector_type: ConnectorType = ConnectorType.MCP,
        headers: dict[str, Any] | None = None,
        authorization: str | None = None,
    ) -> Connector:
        """Register a new connector.

        :param url: URL of the MCP server to connect to.
        :param connector_id: (Optional) User-specified identifier for the connector.
        :param connector_type: (Optional) Type of connector, defaults to MCP.
        :param headers: (Optional) HTTP headers to include when connecting to the server.
        :param authorization: (Optional) OAuth access token for authenticating with the MCP server.
        :returns: The registered Connector.
        """
        # Fetch server info and tools from the MCP server
        # TODO: documentation item: users should be able to pass headers and authorization in the connector input as env variables.
        server_info = await get_mcp_server_info(url, headers=headers, authorization=authorization)
        tools_response = await list_mcp_tools(url, headers=headers, authorization=authorization)

        connector = Connector(
            identifier=server_info.name,
            provider_id="builtin::connectors",
            user_connector_id=connector_id,
            connector_type=connector_type,
            url=url,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            server_name=server_info.name,
            server_label=server_info.title,
            server_description=server_info.description,
            tools=tools_response.data,
        )

        logger.info(f"Registered connector {connector.connector_id} with server name {connector.server_name}")
        self.connectors_map[connector.connector_id] = connector
        return connector

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
        connectors = [c for c in self.connectors_map.values() if registry_id is None or c.registry_id == registry_id]
        if not include_tools:
            return ListConnectorsResponse(data=[c.without_tools for c in connectors])
        return ListConnectorsResponse(data=connectors)

    async def get_connector(self, connector_id: str, include_tools: bool = False) -> Connector:
        """Get a connector by its ID.

        :param connector_id: The ID of the connector to get.
        :returns: A Connector.
        :raises ConnectorNotFoundError: If the connector is not found.
        """
        connector = self.connectors_map.get(connector_id)
        if connector is None:
            raise ConnectorNotFoundError(connector_id)
        if not include_tools:
            return connector.without_tools
        return connector

    async def list_connector_tools(self, connector_id: str) -> ListToolsResponse:
        """List tools available from a connector.

        :param connector_id: The ID of the connector to list tools for.
        :returns: A ListToolsResponse.
        :raises ConnectorNotFoundError: If the connector is not found.
        """
        connector = await self.get_connector(connector_id, include_tools=True)
        # Return empty list if no tools, rather than raising
        return ListToolsResponse(data=connector.tools or [])

    async def get_connector_tool(self, connector_id: str, tool_name: str) -> ToolDef:
        """Get a tool definition by its name from a connector.

        :param connector_id: The ID of the connector to get the tool from.
        :param tool_name: The name of the tool to get.
        :returns: A ToolDef.
        :raises ConnectorNotFoundError: If the connector is not found.
        :raises ConnectorToolNotFoundError: If the tool is not found in the connector.
        """
        connector_tools = await self.list_connector_tools(connector_id)
        for tool in connector_tools.data:
            if tool.name == tool_name:
                return tool
        raise ConnectorToolNotFoundError(connector_id, tool_name)

    async def list_registries(self) -> ListRegistriesResponse:
        """List all registries.

        :returns: A ListRegistriesResponse.
        """
        return ListRegistriesResponse(data=list(self.registries_map.values()))

    async def get_registry(self, registry_id: str) -> Registry:
        """Get a registry by its ID.

        :param registry_id: The ID of the registry to get.
        :returns: A Registry.
        :raises RegistryNotFoundError: If the registry is not found.
        """
        registry = self.registries_map.get(registry_id)
        if registry is None:
            raise RegistryNotFoundError(registry_id)
        return registry

    async def shutdown(self) -> None:
        self.connectors_map.clear()
        self.registries_map.clear()
