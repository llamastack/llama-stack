# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the DatasetIO API.

This module defines the FastAPI router for the DatasetIO API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Path, Query

from llama_stack_api.common.responses import PaginatedResponse
from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1BETA

from .api import DatasetIO
from .models import (
    AppendRowsRequest,
    IterRowsRequest,
)


def create_router(impl: DatasetIO) -> APIRouter:
    """Create a FastAPI router for the DatasetIO API.

    Args:
        impl: The DatasetIO implementation instance

    Returns:
        APIRouter configured for the DatasetIO API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1BETA}",
        tags=["DatasetIO"],
        responses=standard_responses,
    )

    @router.get(
        "/datasetio/iterrows/{dataset_id:path}",
        response_model=PaginatedResponse,
        summary="Get a paginated list of rows from a dataset.",
        description="""Get a paginated list of rows from a dataset.

Uses offset-based pagination where:
- start_index: The starting index (0-based). If None, starts from beginning.
- limit: Number of items to return. If None or -1, returns all items.

The response includes:
- data: List of items for the current page.
- has_more: Whether there are more items available after this set.""",
        responses={
            200: {"description": "A PaginatedResponse containing the rows."},
        },
    )
    async def iterrows(
        dataset_id: Annotated[str, Path(description="The ID of the dataset to get the rows from.")],
        start_index: Annotated[
            int | None, Query(description="Index into dataset for the first row to get. Get all rows if None.")
        ] = None,
        limit: Annotated[int | None, Query(description="The number of rows to get.")] = None,
    ) -> PaginatedResponse:
        request = IterRowsRequest(
            dataset_id=dataset_id,
            start_index=start_index,
            limit=limit,
        )
        return await impl.iterrows(request)

    @router.post(
        "/datasetio/append-rows/{dataset_id:path}",
        status_code=204,
        summary="Append rows to a dataset.",
        description="Append rows to a dataset.",
        responses={
            204: {"description": "Rows were successfully appended."},
        },
    )
    async def append_rows(
        dataset_id: Annotated[str, Path(description="The ID of the dataset to append the rows to.")],
        request: Annotated[AppendRowsRequest, Body(...)],
    ) -> None:
        # Override the dataset_id from the path
        request_with_id = AppendRowsRequest(
            dataset_id=dataset_id,
            rows=request.rows,
        )
        return await impl.append_rows(request_with_id)

    return router
