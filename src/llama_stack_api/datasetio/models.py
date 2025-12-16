# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for DatasetIO API requests and responses.

This module defines the request and response models for the DatasetIO API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api.common.responses import PaginatedResponse
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class IterRowsRequest(BaseModel):
    """Request model for iterating over rows in a dataset."""

    dataset_id: str = Field(..., description="The ID of the dataset to get the rows from.")
    start_index: int | None = Field(
        default=None,
        description="Index into dataset for the first row to get. Get all rows if None.",
    )
    limit: int | None = Field(
        default=None,
        description="The number of rows to get.",
    )


@json_schema_type
class AppendRowsRequest(BaseModel):
    """Request model for appending rows to a dataset."""

    dataset_id: str = Field(..., description="The ID of the dataset to append the rows to.")
    rows: list[dict[str, Any]] = Field(..., description="The rows to append to the dataset.")


__all__ = [
    "AppendRowsRequest",
    "IterRowsRequest",
    "PaginatedResponse",
]
