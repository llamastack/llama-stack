# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import ListShieldsResponse, RegisterShieldRequest, Shield
from .shields_service import ShieldsService


def get_shields_service(request: Request) -> ShieldsService:
    """Dependency to get the shields service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.shields not in impls:
        raise ValueError("Shields API implementation not found")
    return impls[Api.shields]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Shields"],
    responses=standard_responses,
)


@router.get(
    "/shields",
    response_model=ListShieldsResponse,
    summary="List all shields",
    description="List all shields",
)
async def list_shields(svc: ShieldsService = Depends(get_shields_service)) -> ListShieldsResponse:
    """List all shields."""
    return await svc.list_shields()


@router.get(
    "/shields/{identifier:path}",
    response_model=Shield,
    summary="Get a shield by its identifier",
    description="Get a shield by its identifier",
)
async def get_shield(
    identifier: Annotated[str, FastAPIPath(..., description="The identifier of the shield to get")],
    svc: ShieldsService = Depends(get_shields_service),
) -> Shield:
    """Get a shield by its identifier."""
    return await svc.get_shield(identifier=identifier)


@router.post(
    "/shields",
    response_model=Shield,
    summary="Register a shield",
    description="Register a shield",
)
async def register_shield(
    body: RegisterShieldRequest = Body(...),
    svc: ShieldsService = Depends(get_shields_service),
) -> Shield:
    """Register a shield."""
    return await svc.register_shield(
        shield_id=body.shield_id,
        provider_shield_id=body.provider_shield_id,
        provider_id=body.provider_id,
        params=body.params,
    )


@router.delete(
    "/shields/{identifier:path}",
    response_model=None,
    status_code=204,
    summary="Unregister a shield",
    description="Unregister a shield",
)
async def unregister_shield(
    identifier: Annotated[str, FastAPIPath(..., description="The identifier of the shield to unregister")],
    svc: ShieldsService = Depends(get_shields_service),
) -> None:
    """Unregister a shield."""
    await svc.unregister_shield(identifier=identifier)


# For backward compatibility with the router registry system
def create_shields_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Shields API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.shields, create_shields_router)
