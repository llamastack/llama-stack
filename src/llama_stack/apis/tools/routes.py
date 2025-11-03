# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.common.content_types import URL, InterleavedContent
from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import (
    InvokeToolRequest,
    ListToolDefsResponse,
    ListToolGroupsResponse,
    RegisterToolGroupRequest,
    ToolDef,
    ToolGroup,
    ToolInvocationResult,
)
from .rag_tool import RAGDocument, RAGQueryConfig, RAGQueryResult
from .tool_groups_service import ToolGroupsService
from .tool_runtime_service import ToolRuntimeService


def get_tool_groups_service(request: Request) -> ToolGroupsService:
    """Dependency to get the tool groups service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.tool_groups not in impls:
        raise ValueError("Tool Groups API implementation not found")
    return impls[Api.tool_groups]


def get_tool_runtime_service(request: Request) -> ToolRuntimeService:
    """Dependency to get the tool runtime service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.tool_runtime not in impls:
        raise ValueError("Tool Runtime API implementation not found")
    return impls[Api.tool_runtime]


# Tool Groups Router
tool_groups_router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Tool Groups"],
    responses=standard_responses,
)


@tool_groups_router.post(
    "/toolgroups",
    response_model=None,
    status_code=204,
    summary="Register a tool group",
    description="Register a tool group",
)
async def register_tool_group(
    body: RegisterToolGroupRequest = Body(...),
    svc: ToolGroupsService = Depends(get_tool_groups_service),
) -> None:
    """Register a tool group."""
    await svc.register_tool_group(
        toolgroup_id=body.toolgroup_id,
        provider_id=body.provider_id,
        mcp_endpoint=body.mcp_endpoint,
        args=body.args,
    )


@tool_groups_router.get(
    "/toolgroups/{toolgroup_id:path}",
    response_model=ToolGroup,
    summary="Get a tool group by its ID",
    description="Get a tool group by its ID",
)
async def get_tool_group(
    toolgroup_id: Annotated[str, FastAPIPath(..., description="The ID of the tool group to get")],
    svc: ToolGroupsService = Depends(get_tool_groups_service),
) -> ToolGroup:
    """Get a tool group by its ID."""
    return await svc.get_tool_group(toolgroup_id=toolgroup_id)


@tool_groups_router.get(
    "/toolgroups",
    response_model=ListToolGroupsResponse,
    summary="List tool groups",
    description="List tool groups with optional provider",
)
async def list_tool_groups(svc: ToolGroupsService = Depends(get_tool_groups_service)) -> ListToolGroupsResponse:
    """List tool groups."""
    return await svc.list_tool_groups()


@tool_groups_router.get(
    "/tools",
    response_model=ListToolDefsResponse,
    summary="List tools",
    description="List tools with optional tool group",
)
async def list_tools(
    toolgroup_id: str | None = Query(None, description="The ID of the tool group to list tools for"),
    svc: ToolGroupsService = Depends(get_tool_groups_service),
) -> ListToolDefsResponse:
    """List tools."""
    return await svc.list_tools(toolgroup_id=toolgroup_id)


@tool_groups_router.get(
    "/tools/{tool_name:path}",
    response_model=ToolDef,
    summary="Get a tool by its name",
    description="Get a tool by its name",
)
async def get_tool(
    tool_name: Annotated[str, FastAPIPath(..., description="The name of the tool to get")],
    svc: ToolGroupsService = Depends(get_tool_groups_service),
) -> ToolDef:
    """Get a tool by its name."""
    return await svc.get_tool(tool_name=tool_name)


@tool_groups_router.delete(
    "/toolgroups/{toolgroup_id:path}",
    response_model=None,
    status_code=204,
    summary="Unregister a tool group",
    description="Unregister a tool group",
)
async def unregister_toolgroup(
    toolgroup_id: Annotated[str, FastAPIPath(..., description="The ID of the tool group to unregister")],
    svc: ToolGroupsService = Depends(get_tool_groups_service),
) -> None:
    """Unregister a tool group."""
    await svc.unregister_toolgroup(toolgroup_id=toolgroup_id)


# Tool Runtime Router
tool_runtime_router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Tool Runtime"],
    responses=standard_responses,
)


@tool_runtime_router.get(
    "/tool-runtime/list-tools",
    response_model=ListToolDefsResponse,
    summary="List all tools in the runtime",
    description="List all tools in the runtime",
)
async def list_runtime_tools(
    tool_group_id: str | None = Query(None, description="The ID of the tool group to list tools for"),
    mcp_endpoint: str | None = Query(None, description="The MCP endpoint URL to use for the tool group"),
    svc: ToolRuntimeService = Depends(get_tool_runtime_service),
) -> ListToolDefsResponse:
    """List all tools in the runtime."""
    url_obj = URL(uri=mcp_endpoint) if mcp_endpoint else None
    return await svc.list_runtime_tools(tool_group_id=tool_group_id, mcp_endpoint=url_obj)


@tool_runtime_router.post(
    "/tool-runtime/invoke",
    response_model=ToolInvocationResult,
    summary="Run a tool with the given arguments",
    description="Run a tool with the given arguments",
)
async def invoke_tool(
    body: InvokeToolRequest = Body(...),
    svc: ToolRuntimeService = Depends(get_tool_runtime_service),
) -> ToolInvocationResult:
    """Invoke a tool."""
    return await svc.invoke_tool(tool_name=body.tool_name, kwargs=body.kwargs)


@tool_runtime_router.post(
    "/tool-runtime/rag-tool/insert",
    response_model=None,
    status_code=204,
    summary="Insert documents into the RAG system.",
    description="Index documents so they can be used by the RAG system.",
)
async def rag_tool_insert(
    documents: list[RAGDocument] = Body(..., description="List of documents to index in the RAG system."),
    vector_store_id: str = Body(..., description="ID of the vector database to store the document embeddings."),
    chunk_size_in_tokens: int = Body(512, description="Size in tokens for document chunking during indexing."),
    svc: ToolRuntimeService = Depends(get_tool_runtime_service),
) -> None:
    """Insert documents into the RAG system."""
    if svc.rag_tool is None:
        raise ValueError("RAG tool is not available")
    await svc.rag_tool.insert(
        documents=documents,
        vector_store_id=vector_store_id,
        chunk_size_in_tokens=chunk_size_in_tokens,
    )


@tool_runtime_router.post(
    "/tool-runtime/rag-tool/query",
    response_model=RAGQueryResult,
    summary="Query the RAG system for context.",
    description="Query the RAG system for context; typically invoked by the agent.",
)
async def rag_tool_query(
    content: InterleavedContent = Body(..., description="The query content to search for in the indexed documents."),
    vector_store_ids: list[str] = Body(..., description="List of vector database IDs to search within."),
    query_config: RAGQueryConfig | None = Body(None, description="Configuration parameters for the query operation."),
    svc: ToolRuntimeService = Depends(get_tool_runtime_service),
) -> RAGQueryResult:
    """Query the RAG system for context."""
    if svc.rag_tool is None:
        raise ValueError("RAG tool is not available")
    return await svc.rag_tool.query(
        content=content,
        vector_store_ids=vector_store_ids,
        query_config=query_config,
    )


# For backward compatibility with the router registry system
def create_tool_groups_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Tool Groups API (legacy compatibility)."""
    return tool_groups_router


def create_tool_runtime_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Tool Runtime API (legacy compatibility)."""
    return tool_runtime_router


# Register the router factories
register_router(Api.tool_groups, create_tool_groups_router)
register_router(Api.tool_runtime, create_tool_runtime_router)
