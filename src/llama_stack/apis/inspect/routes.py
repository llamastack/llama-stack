# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from fastapi import Depends, Request

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .inspect_service import InspectService
from .models import HealthInfo, ListRoutesResponse, VersionInfo


def get_inspect_service(request: Request) -> InspectService:
    """Dependency to get the inspect service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.inspect not in impls:
        raise ValueError("Inspect API implementation not found")
    return impls[Api.inspect]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Inspect"],
    responses=standard_responses,
)


@router.get(
    "/inspect/routes",
    response_model=ListRoutesResponse,
    summary="List routes.",
    description="List all available API routes with their methods and implementing providers.",
)
async def list_routes(svc: InspectService = Depends(get_inspect_service)) -> ListRoutesResponse:
    """List all available API routes."""
    return await svc.list_routes()


@router.get(
    "/health",
    response_model=HealthInfo,
    summary="Get health status.",
    description="Get the current health status of the service.",
)
async def health(svc: InspectService = Depends(get_inspect_service)) -> HealthInfo:
    """Get the current health status of the service."""
    return await svc.health()


@router.get(
    "/version",
    response_model=VersionInfo,
    summary="Get version.",
    description="Get the version of the service.",
)
async def version(svc: InspectService = Depends(get_inspect_service)) -> VersionInfo:
    """Get the version of the service."""
    return await svc.version()


# For backward compatibility with the router registry system
def create_inspect_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Inspect API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.inspect, create_inspect_router)
