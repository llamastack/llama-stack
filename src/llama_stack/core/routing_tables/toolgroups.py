# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import AuthenticationRequiredError, ToolGroupWithOwner
from llama_stack.log import get_logger
from llama_stack_api import (
    URL,
    ListToolDefsResponse,
    ListToolGroupsResponse,
    ToolDef,
    ToolGroup,
    ToolGroupNotFoundError,
    ToolGroups,
)

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


def parse_toolgroup_from_toolgroup_name_pair(toolgroup_name_with_maybe_tool_name: str) -> str | None:
    # handle the funny case like "builtin::rag/knowledge_search"
    parts = toolgroup_name_with_maybe_tool_name.split("/")
    if len(parts) == 2:
        return parts[0]
    else:
        return None


class ToolGroupsRoutingTable(CommonRoutingTableImpl, ToolGroups):
    toolgroups_to_tools: dict[str, list[ToolDef]] = {}
    tool_to_toolgroup: dict[str, str] = {}

    # overridden
    async def get_provider_impl(self, routing_key: str, provider_id: str | None = None) -> Any:
        # we don't index tools in the registry anymore, but only keep a cache of them by toolgroup_id
        # TODO: we may want to invalidate the cache (for a given toolgroup_id) every once in a while?

        toolgroup_id = parse_toolgroup_from_toolgroup_name_pair(routing_key)
        if toolgroup_id:
            routing_key = toolgroup_id

        if routing_key in self.tool_to_toolgroup:
            routing_key = self.tool_to_toolgroup[routing_key]
        return await super().get_provider_impl(routing_key, provider_id)

    async def list_tools(
        self, toolgroup_id: str | None = None, authorization: str | None = None
    ) -> ListToolDefsResponse:
        if toolgroup_id:
            if group_id := parse_toolgroup_from_toolgroup_name_pair(toolgroup_id):
                toolgroup_id = group_id
            toolgroups = [await self.get_tool_group(toolgroup_id)]
        else:
            toolgroups = await self.get_all_with_type("tool_group")

        all_tools = []
        for toolgroup in toolgroups:
            if toolgroup.identifier not in self.toolgroups_to_tools:
                try:
                    await self._index_tools(toolgroup, authorization=authorization)
                except AuthenticationRequiredError:
                    # Send authentication errors back to the client so it knows
                    # that it needs to supply credentials for remote MCP servers.
                    raise
                except Exception as e:
                    # Other errors that the client cannot fix are logged and
                    # those specific toolgroups are skipped.
                    logger.warning(f"Error listing tools for toolgroup {toolgroup.identifier}: {e}")
                    logger.debug(e, exc_info=True)
                    continue
            all_tools.extend(self.toolgroups_to_tools[toolgroup.identifier])

        return ListToolDefsResponse(data=all_tools)

    async def _index_tools(self, toolgroup: ToolGroup, authorization: str | None = None):
        provider_impl = await super().get_provider_impl(toolgroup.identifier, toolgroup.provider_id)

        tooldefs_response = await provider_impl.list_runtime_tools(
            toolgroup.identifier, toolgroup.mcp_endpoint, authorization=authorization
        )
        tooldefs = tooldefs_response.data

        for t in tooldefs:
            t.toolgroup_id = toolgroup.identifier
            # Include MCP server instructions in metadata for easy access
            if toolgroup.mcp_server_instructions:
                if t.metadata is None:
                    t.metadata = {}
                t.metadata["mcp_server_instructions"] = toolgroup.mcp_server_instructions

        self.toolgroups_to_tools[toolgroup.identifier] = tooldefs
        for tool in tooldefs:
            self.tool_to_toolgroup[tool.name] = toolgroup.identifier

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        return ListToolGroupsResponse(data=await self.get_all_with_type("tool_group"))

    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup:
        tool_group = await self.get_object_by_identifier("tool_group", toolgroup_id)
        if tool_group is None:
            raise ToolGroupNotFoundError(toolgroup_id)
        return tool_group

    async def get_tool(self, tool_name: str) -> ToolDef:
        if tool_name in self.tool_to_toolgroup:
            toolgroup_id = self.tool_to_toolgroup[tool_name]
            tools = self.toolgroups_to_tools[toolgroup_id]
            for tool in tools:
                if tool.name == tool_name:
                    return tool
        raise ValueError(f"Tool '{tool_name}' not found")

    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
        mcp_server_instructions: str | None = None,
    ) -> None:
        # If this is an MCP toolgroup and we don't have instructions yet, try to fetch them
        if mcp_endpoint and not mcp_server_instructions:
            try:
                from llama_stack.providers.utils.tools.mcp import list_mcp_tools, prepare_mcp_headers

                headers = args.get("headers") if args else None
                final_headers = prepare_mcp_headers(headers, authorization=None)

                tools_response = await list_mcp_tools(
                    endpoint=str(mcp_endpoint),
                    headers=final_headers,
                    authorization=None,
                )
                # Extract instructions from the first tool's metadata (all tools share same server instructions)
                if tools_response.data:
                    first_tool = tools_response.data[0]
                    if first_tool.metadata and "mcp_server_instructions" in first_tool.metadata:
                        mcp_server_instructions = first_tool.metadata["mcp_server_instructions"]
            except Exception as e:
                logger.warning(f"Failed to fetch MCP server instructions for {toolgroup_id}: {e}")
                # Continue without instructions

        toolgroup = ToolGroupWithOwner(
            identifier=toolgroup_id,
            provider_id=provider_id,
            provider_resource_id=toolgroup_id,
            mcp_endpoint=mcp_endpoint,
            args=args,
            mcp_server_instructions=mcp_server_instructions,
        )
        await self.register_object(toolgroup)

        # ideally, indexing of the tools should not be necessary because anyone using
        # the tools should first list the tools and then use them. but there are assumptions
        # baked in some of the code and tests right now.
        if not toolgroup.mcp_endpoint:
            await self._index_tools(toolgroup)

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        await self.unregister_object(await self.get_tool_group(toolgroup_id))

    async def shutdown(self) -> None:
        pass
