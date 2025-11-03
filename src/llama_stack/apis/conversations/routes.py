# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .conversations_service import ConversationService
from .models import (
    Conversation,
    ConversationCreateRequest,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemCreateRequest,
    ConversationItemDeletedResource,
    ConversationItemInclude,
    ConversationItemList,
    ConversationUpdateRequest,
)


def get_conversation_service(request: Request) -> ConversationService:
    """Dependency to get the conversation service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.conversations not in impls:
        raise ValueError("Conversations API implementation not found")
    return impls[Api.conversations]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Conversations"],
    responses=standard_responses,
)


@router.post(
    "/conversations",
    response_model=Conversation,
    summary="Create a conversation",
    description="Create a conversation",
)
async def create_conversation(
    body: ConversationCreateRequest = Body(...),
    svc: ConversationService = Depends(get_conversation_service),
) -> Conversation:
    """Create a conversation."""
    return await svc.create_conversation(items=body.items, metadata=body.metadata)


@router.get(
    "/conversations/{conversation_id}",
    response_model=Conversation,
    summary="Retrieve a conversation",
    description="Get a conversation with the given ID",
)
async def get_conversation(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    svc: ConversationService = Depends(get_conversation_service),
) -> Conversation:
    """Get a conversation."""
    return await svc.get_conversation(conversation_id=conversation_id)


@router.post(
    "/conversations/{conversation_id}",
    response_model=Conversation,
    summary="Update a conversation",
    description="Update a conversation's metadata with the given ID",
)
async def update_conversation(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    body: ConversationUpdateRequest = Body(...),
    svc: ConversationService = Depends(get_conversation_service),
) -> Conversation:
    """Update a conversation."""
    return await svc.update_conversation(conversation_id=conversation_id, metadata=body.metadata)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ConversationDeletedResource,
    summary="Delete a conversation",
    description="Delete a conversation with the given ID",
)
async def openai_delete_conversation(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    svc: ConversationService = Depends(get_conversation_service),
) -> ConversationDeletedResource:
    """Delete a conversation."""
    return await svc.openai_delete_conversation(conversation_id=conversation_id)


@router.post(
    "/conversations/{conversation_id}/items",
    response_model=ConversationItemList,
    summary="Create items",
    description="Create items in the conversation",
)
async def add_items(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    body: ConversationItemCreateRequest = Body(...),
    svc: ConversationService = Depends(get_conversation_service),
) -> ConversationItemList:
    """Create items in the conversation."""
    return await svc.add_items(conversation_id=conversation_id, items=body.items)


@router.get(
    "/conversations/{conversation_id}/items/{item_id}",
    response_model=ConversationItem,
    summary="Retrieve an item",
    description="Retrieve a conversation item",
)
async def retrieve(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    item_id: Annotated[str, FastAPIPath(..., description="The item identifier")],
    svc: ConversationService = Depends(get_conversation_service),
) -> ConversationItem:
    """Retrieve a conversation item."""
    return await svc.retrieve(conversation_id=conversation_id, item_id=item_id)


@router.get(
    "/conversations/{conversation_id}/items",
    response_model=ConversationItemList,
    summary="List items",
    description="List items in the conversation",
)
async def list_items(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    after: str | None = Query(None, description="An item ID to list items after, used in pagination"),
    include: list[ConversationItemInclude] | None = Query(
        None, description="Specify additional output data to include in the response"
    ),
    limit: int | None = Query(None, description="A limit on the number of objects to be returned (1-100, default 20)"),
    order: Literal["asc", "desc"] | None = Query(
        None, description="The order to return items in (asc or desc, default desc)"
    ),
    svc: ConversationService = Depends(get_conversation_service),
) -> ConversationItemList:
    """List items in the conversation."""
    return await svc.list_items(conversation_id=conversation_id, after=after, include=include, limit=limit, order=order)


@router.delete(
    "/conversations/{conversation_id}/items/{item_id}",
    response_model=ConversationItemDeletedResource,
    summary="Delete an item",
    description="Delete a conversation item",
)
async def openai_delete_conversation_item(
    conversation_id: Annotated[str, FastAPIPath(..., description="The conversation identifier")],
    item_id: Annotated[str, FastAPIPath(..., description="The item identifier")],
    svc: ConversationService = Depends(get_conversation_service),
) -> ConversationItemDeletedResource:
    """Delete a conversation item."""
    return await svc.openai_delete_conversation_item(conversation_id=conversation_id, item_id=item_id)


# For backward compatibility with the router registry system
def create_conversations_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Conversations API (legacy compatibility)."""
    return router


# Register the router factory
register_router(Api.conversations, create_conversations_router)
