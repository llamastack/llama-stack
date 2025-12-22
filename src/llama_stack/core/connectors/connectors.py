# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import cast

from pydantic import BaseModel

from llama_stack.core.datatypes import StackConfig
from llama_stack.core.storage.kvstore import KVStore, kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.tools.mcp import get_mcp_server_info, list_mcp_tools
from llama_stack_api import (
    Connector,
    ConnectorNotFoundError,
    Connectors,
    ConnectorSource,
    ConnectorToolNotFoundError,
    ConnectorType,
    ListConnectorsResponse,
    ListToolsResponse,
    ToolDef,
)

logger = get_logger(name=__name__, category="connectors")


class ConnectorServiceConfig(BaseModel):
    """Configuration for the built-in connector service.
    :param run_config: Stack run configuration for resolving persistence
    """

    config: StackConfig


async def get_provider_impl(config: ConnectorServiceConfig):
    """Get the connector service implementation."""
    impl = ConnectorServiceImpl(config)
    return impl


KEY_PREFIX = "connectors:v1:"


class ConnectorServiceImpl(Connectors):
    """Built-in connector service implementation."""

    def __init__(self, config: ConnectorServiceConfig):
        self.config = config
        self.kvstore: KVStore

    def _get_key(self, connector_id: str) -> str:
        """Get the KVStore key for a connector."""
        return f"{KEY_PREFIX}{connector_id}"

    async def initialize(self):
        """Initialize the connector service."""

        # Use connectors store reference from run config
        connectors_ref = self.config.config.storage.stores.connectors
        if not connectors_ref:
            raise ValueError("storage.stores.connectors must be configured in run config")
        self.kvstore = await kvstore_impl(connectors_ref)

    async def register_connector(
        self,
        connector_id: str,
        connector_type: ConnectorType,
        source: ConnectorSource,
        url: str,
        server_label: str | None = None,
        server_name: str | None = None,
        server_description: str | None = None,
    ) -> Connector:
        """Register a new connector (idempotent - updates if already exists)."""

        connector = Connector(
            connector_id=connector_id,
            connector_type=connector_type,
            source=source,
            url=url,
            server_label=server_label,
            server_name=server_name,
            server_description=server_description,
        )

        key = self._get_key(connector_id)
        existing_connector_json = await self.kvstore.get(key)

        if existing_connector_json:
            existing_connector = Connector.model_validate_json(existing_connector_json)

            # Only overwrite if the connector is an exact match; otherwise log and keep existing.
            if existing_connector.model_dump() != connector.model_dump():
                logger.info(
                    "Connector %s already exists with different configuration; skipping overwrite",
                    connector_id,
                )
                return existing_connector

            logger.debug("Connector %s already exists and matches; overwriting with same value", connector_id)

        # Persist full connector, including source (Field is excluded from schema/dumps by default).
        connector_payload = connector.model_dump()
        connector_payload["source"] = connector.source
        await self.kvstore.set(key, json.dumps(connector_payload))

        return connector

    async def unregister_connector(self, connector_id: str):
        """Unregister a connector."""
        key = self._get_key(connector_id)
        if not await self.kvstore.get(key):
            return
        await self.kvstore.delete(key)

    async def list_connectors(self, source: ConnectorSource | None = None) -> ListConnectorsResponse:
        """List all registered connectors.

        :param source: (Optional) Source of the connectors to list.
        :returns: A ListConnectorsResponse.
        """

        connectors: list[Connector] = []
        # Get all keys with the connector prefix
        keys = await self.kvstore.keys_in_range(KEY_PREFIX, KEY_PREFIX + "\uffff")
        for key in keys:
            connector_json = await self.kvstore.get(key)
            if not connector_json:
                continue
            connector = Connector.model_validate_json(connector_json)
            if source is not None and connector.source != source:
                continue
            connectors.append(connector)
        return ListConnectorsResponse(data=connectors)

    async def get_connector(
        self,
        connector_id: str,
        authorization: str | None = None,
    ) -> Connector:
        """Get a connector by its ID."""

        connector_json = await self.kvstore.get(self._get_key(connector_id))
        if not connector_json:
            raise ConnectorNotFoundError(connector_id)
        connector = Connector.model_validate_json(connector_json)

        server_info = await get_mcp_server_info(connector.url, authorization=authorization)
        connector.server_name = server_info.name
        connector.server_description = server_info.description
        connector.server_version = server_info.version
        return connector

    async def list_connector_tools(
        self,
        connector_id: str,
        authorization: str | None = None,
    ) -> ListToolsResponse:
        """List all tools available from a connector."""

        connector = await self.get_connector(connector_id, authorization=authorization)
        tools_response = await list_mcp_tools(connector.url, authorization=authorization)
        return ListToolsResponse(data=tools_response.data)

    async def get_connector_tool(
        self,
        connector_id: str,
        tool_name: str,
        authorization: str | None = None,
    ) -> ToolDef:
        """Get a tool by its name from a connector."""

        connector = await self.get_connector(connector_id, authorization=authorization)
        tools_response = await list_mcp_tools(connector.url, authorization=authorization)
        for tool in tools_response.data:
            if tool.name == tool_name:
                return cast(ToolDef, tool)
        raise ConnectorToolNotFoundError(connector_id, tool_name)

    async def shutdown(self):
        """Shutdown the connector service."""
        await self.kvstore.close()
