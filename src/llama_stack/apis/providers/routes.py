# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Depends, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import ListProvidersResponse, ProviderInfo
from .providers_service import ProviderService


def get_provider_service(request: Request) -> ProviderService:
    """Dependency to get the provider service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.providers not in impls:
        raise ValueError("Providers API implementation not found")
    return impls[Api.providers]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Providers"],
    responses=standard_responses,
)


@router.get(
    "/providers",
    response_model=ListProvidersResponse,
    summary="List providers",
    description="List all available providers",
)
async def list_providers(svc: ProviderService = Depends(get_provider_service)) -> ListProvidersResponse:
    """List all available providers."""
    return await svc.list_providers()


@router.get(
    "/providers/{provider_id}",
    response_model=ProviderInfo,
    summary="Get provider",
    description="Get detailed information about a specific provider",
)
async def inspect_provider(
    provider_id: Annotated[str, FastAPIPath(..., description="The ID of the provider to inspect")],
    svc: ProviderService = Depends(get_provider_service),
) -> ProviderInfo:
    """Get detailed information about a specific provider."""
    return await svc.inspect_provider(provider_id=provider_id)


# For backward compatibility with the router registry system
def create_providers_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Providers API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.providers, create_providers_router)
