# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Agents API.

This module defines the FastAPI router for the Agents API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query

from llama_stack_api.common.responses import Order
from llama_stack_api.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
)
from llama_stack_api.router_utils import (
    create_path_dependency,
    create_query_dependency,
    standard_responses,
)
from llama_stack_api.version import LLAMA_STACK_API_V1

from .api import Agents
from .models import (
    CreateResponseRequest,
    DeleteResponseRequest,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    RetrieveResponseRequest,
)

# Automatically generate dependency functions from Pydantic models
get_retrieve_response_request = create_path_dependency(RetrieveResponseRequest)
get_delete_response_request = create_path_dependency(DeleteResponseRequest)
get_list_responses_request = create_query_dependency(ListResponsesRequest)


# Manual dependency for ListResponseInputItemsRequest since it mixes Path and Query parameters
async def get_list_response_input_items_request(
    response_id: Annotated[str, Path(description="The ID of the response to retrieve input items for.")],
    after: Annotated[
        str | None,
        Query(description="An item ID to list items after, used for pagination."),
    ] = None,
    before: Annotated[
        str | None,
        Query(description="An item ID to list items before, used for pagination."),
    ] = None,
    include: Annotated[
        list[str] | None,
        Query(description="Additional fields to include in the response."),
    ] = None,
    limit: Annotated[
        int | None,
        Query(
            description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20."
        ),
    ] = 20,
    order: Annotated[Order | None, Query(description="The order to return the input items in.")] = Order.desc,
) -> ListResponseInputItemsRequest:
    return ListResponseInputItemsRequest(
        response_id=response_id,
        after=after,
        before=before,
        include=include,
        limit=limit,
        order=order,
    )


def create_router(impl: Agents) -> APIRouter:
    """Create a FastAPI router for the Agents API.

    Args:
        impl: The Agents implementation instance

    Returns:
        APIRouter configured for the Agents API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1}",
        tags=["Agents"],
        responses=standard_responses,
    )

    @router.get(
        "/responses/{response_id}",
        response_model=OpenAIResponseObject,
        summary="Get a model response.",
        description="Get a model response.",
    )
    async def get_openai_response(
        request: Annotated[RetrieveResponseRequest, Depends(get_retrieve_response_request)],
    ) -> OpenAIResponseObject:
        return await impl.get_openai_response(request)

    @router.post(
        "/responses",
        summary="Create a model response.",
        description="Create a model response.",
        status_code=200,
        responses={
            200: {
                "description": "An OpenAIResponseObject or a stream of OpenAIResponseObjectStream.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/OpenAIResponseObject"}},
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/OpenAIResponseObjectStream"}},
                },
            }
        },
    )
    async def create_openai_response(
        request: Annotated[CreateResponseRequest, Body(...)],
    ):
        # Return type annotation omitted on purpose because it can be a stream
        return await impl.create_openai_response(request)

    @router.get(
        "/responses",
        response_model=ListOpenAIResponseObject,
        summary="List all responses.",
        description="List all responses.",
    )
    async def list_openai_responses(
        request: Annotated[ListResponsesRequest, Depends(get_list_responses_request)],
    ) -> ListOpenAIResponseObject:
        return await impl.list_openai_responses(request)

    @router.get(
        "/responses/{response_id}/input_items",
        response_model=ListOpenAIResponseInputItem,
        summary="List input items.",
        description="List input items.",
    )
    async def list_openai_response_input_items(
        request: Annotated[ListResponseInputItemsRequest, Depends(get_list_response_input_items_request)],
    ) -> ListOpenAIResponseInputItem:
        return await impl.list_openai_response_input_items(request)

    @router.delete(
        "/responses/{response_id}",
        response_model=OpenAIDeleteResponseObject,
        summary="Delete a response.",
        description="Delete a response.",
    )
    async def delete_openai_response(
        request: Annotated[DeleteResponseRequest, Depends(get_delete_response_request)],
    ) -> OpenAIDeleteResponseObject:
        return await impl.delete_openai_response(request)

    return router
