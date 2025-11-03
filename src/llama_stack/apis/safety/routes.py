# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from fastapi import Body, Depends, Request

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import ModerationObject, RunModerationRequest, RunShieldRequest, RunShieldResponse
from .safety_service import SafetyService


def get_safety_service(request: Request) -> SafetyService:
    """Dependency to get the safety service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.safety not in impls:
        raise ValueError("Safety API implementation not found")
    return impls[Api.safety]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Safety"],
    responses=standard_responses,
)


@router.post(
    "/safety/run-shield",
    response_model=RunShieldResponse,
    summary="Run shield.",
    description="Run a shield.",
)
async def run_shield(
    body: RunShieldRequest = Body(...),
    svc: SafetyService = Depends(get_safety_service),
) -> RunShieldResponse:
    """Run a shield."""
    return await svc.run_shield(shield_id=body.shield_id, messages=body.messages, params=body.params)


@router.post(
    "/moderations",
    response_model=ModerationObject,
    summary="Create moderation.",
    description="Classifies if text and/or image inputs are potentially harmful.",
)
async def run_moderation(
    body: RunModerationRequest = Body(...),
    svc: SafetyService = Depends(get_safety_service),
) -> ModerationObject:
    """Create moderation."""
    return await svc.run_moderation(input=body.input, model=body.model)


# For backward compatibility with the router registry system
def create_safety_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Safety API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.safety, create_safety_router)
