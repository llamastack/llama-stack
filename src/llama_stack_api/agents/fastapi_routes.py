# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Agents API.

This module defines the FastAPI router for the Agents API using standard
FastAPI route decorators.
"""

import asyncio
import contextvars
import json
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
    ResponseItemInclude,
    RetrieveResponseRequest,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "agents"})


def create_sse_event(data: Any) -> str:
    """Create a Server-Sent Event string from data."""
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)
    return f"data: {data}\n\n"


async def sse_generator(event_gen):
    """Convert an async generator to SSE format.

    This function iterates over an async generator and formats each yielded
    item as a Server-Sent Event.
    """
    try:
        async for item in event_gen:
            yield create_sse_event(item)
    except asyncio.CancelledError:
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise  # Re-raise to maintain proper cancellation semantics
    except Exception as e:
        logger.exception("Error in SSE generator")
        exc = _http_exception_from_sse_error(e)
        yield create_sse_event({"error": {"status_code": exc.status_code, "message": exc.detail}})


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
        list[ResponseItemInclude] | None,
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


def _http_exception_from_value_error(exc: ValueError) -> HTTPException:
    """Convert implementation `ValueError` into an OpenAI-compatible HTTP error.

    The compatibility OpenAI client maps HTTP 400 -> `BadRequestError`.
    The existing API surface (and integration tests) expect "not found" cases
    to be represented as a 400, not a 404.
    """

    detail = str(exc) or "Invalid value"
    return HTTPException(status_code=400, detail=detail)


def _http_exception_from_sse_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return _http_exception_from_value_error(exc)
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal server error: An unexpected error occurred.")


def _preserve_context_for_sse(event_gen):
    # StreamingResponse runs in a different task, losing request contextvars.
    # create_task inside context.run captures the context at task creation.
    context = contextvars.copy_context()

    async def wrapper():
        try:
            while True:
                try:
                    task = context.run(asyncio.create_task, event_gen.__anext__())
                    item = await task
                except StopAsyncIteration:
                    break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            if hasattr(event_gen, "aclose"):
                await event_gen.aclose()
            raise

    return wrapper()


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
        try:
            return await impl.get_openai_response(request)
        except ValueError as exc:
            raise _http_exception_from_value_error(exc) from exc

    @router.post(
        "/responses",
        summary="Create a model response.",
        description="Create a model response.",
        status_code=200,
        response_model=None,
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
    ) -> OpenAIResponseObject | StreamingResponse:
        try:
            result = await impl.create_openai_response(request)
        except ValueError as exc:
            raise _http_exception_from_value_error(exc) from exc

        # For streaming responses, wrap in StreamingResponse for HTTP requests.
        # The implementation is typed to return an `AsyncIterator` for streaming.
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(sse_generator(result)),
                media_type="text/event-stream",
            )

        return result

    @router.get(
        "/responses",
        response_model=ListOpenAIResponseObject,
        summary="List all responses.",
        description="List all responses.",
    )
    async def list_openai_responses(
        request: Annotated[ListResponsesRequest, Depends(get_list_responses_request)],
    ) -> ListOpenAIResponseObject:
        try:
            return await impl.list_openai_responses(request)
        except ValueError as exc:
            raise _http_exception_from_value_error(exc) from exc

    @router.get(
        "/responses/{response_id}/input_items",
        response_model=ListOpenAIResponseInputItem,
        summary="List input items.",
        description="List input items.",
    )
    async def list_openai_response_input_items(
        request: Annotated[ListResponseInputItemsRequest, Depends(get_list_response_input_items_request)],
    ) -> ListOpenAIResponseInputItem:
        try:
            return await impl.list_openai_response_input_items(request)
        except ValueError as exc:
            raise _http_exception_from_value_error(exc) from exc

    @router.delete(
        "/responses/{response_id}",
        response_model=OpenAIDeleteResponseObject,
        summary="Delete a response.",
        description="Delete a response.",
    )
    async def delete_openai_response(
        request: Annotated[DeleteResponseRequest, Depends(get_delete_response_request)],
    ) -> OpenAIDeleteResponseObject:
        try:
            return await impl.delete_openai_response(request)
        except ValueError as exc:
            raise _http_exception_from_value_error(exc) from exc

    return router
