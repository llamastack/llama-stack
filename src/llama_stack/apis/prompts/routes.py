# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .models import (
    CreatePromptRequest,
    ListPromptsResponse,
    Prompt,
    SetDefaultVersionRequest,
    UpdatePromptRequest,
)
from .prompts_service import PromptService


def get_prompt_service(request: Request) -> PromptService:
    """Dependency to get the prompt service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.prompts not in impls:
        raise ValueError("Prompts API implementation not found")
    return impls[Api.prompts]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Prompts"],
    responses=standard_responses,
)


@router.get(
    "/prompts",
    response_model=ListPromptsResponse,
    summary="List all prompts",
    description="List all prompts registered in Llama Stack",
)
async def list_prompts(svc: PromptService = Depends(get_prompt_service)) -> ListPromptsResponse:
    """List all prompts."""
    return await svc.list_prompts()


@router.get(
    "/prompts/{prompt_id}/versions",
    response_model=ListPromptsResponse,
    summary="List prompt versions",
    description="List all versions of a specific prompt",
)
async def list_prompt_versions(
    prompt_id: Annotated[str, FastAPIPath(..., description="The identifier of the prompt to list versions for")],
    svc: PromptService = Depends(get_prompt_service),
) -> ListPromptsResponse:
    """List prompt versions."""
    return await svc.list_prompt_versions(prompt_id=prompt_id)


@router.get(
    "/prompts/{prompt_id}",
    response_model=Prompt,
    summary="Get prompt",
    description="Get a prompt by its identifier and optional version",
)
async def get_prompt(
    prompt_id: Annotated[str, FastAPIPath(..., description="The identifier of the prompt to get")],
    version: int | None = Query(None, description="The version of the prompt to get (defaults to latest)"),
    svc: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Get prompt by its identifier and optional version."""
    return await svc.get_prompt(prompt_id=prompt_id, version=version)


@router.post(
    "/prompts",
    response_model=Prompt,
    summary="Create prompt",
    description="Create a new prompt",
)
async def create_prompt(
    body: CreatePromptRequest = Body(...),
    svc: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Create a new prompt."""
    return await svc.create_prompt(prompt=body.prompt, variables=body.variables)


@router.post(
    "/prompts/{prompt_id}",
    response_model=Prompt,
    summary="Update prompt",
    description="Update an existing prompt (increments version)",
)
async def update_prompt(
    prompt_id: Annotated[str, FastAPIPath(..., description="The identifier of the prompt to update")],
    body: UpdatePromptRequest = Body(...),
    svc: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Update an existing prompt."""
    return await svc.update_prompt(
        prompt_id=prompt_id,
        prompt=body.prompt,
        version=body.version,
        variables=body.variables,
        set_as_default=body.set_as_default,
    )


@router.delete(
    "/prompts/{prompt_id}",
    response_model=None,
    status_code=204,
    summary="Delete prompt",
    description="Delete a prompt",
)
async def delete_prompt(
    prompt_id: Annotated[str, FastAPIPath(..., description="The identifier of the prompt to delete")],
    svc: PromptService = Depends(get_prompt_service),
) -> None:
    """Delete a prompt."""
    await svc.delete_prompt(prompt_id=prompt_id)


@router.post(
    "/prompts/{prompt_id}/set-default-version",
    response_model=Prompt,
    summary="Set prompt version",
    description="Set which version of a prompt should be the default in get_prompt (latest)",
)
async def set_default_version(
    prompt_id: Annotated[str, FastAPIPath(..., description="The identifier of the prompt")],
    body: SetDefaultVersionRequest = Body(...),
    svc: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Set which version of a prompt should be the default."""
    return await svc.set_default_version(prompt_id=prompt_id, version=body.version)


# For backward compatibility with the router registry system
def create_prompts_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Prompts API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.prompts, create_prompts_router)
