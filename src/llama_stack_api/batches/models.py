# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Batches API requests and responses.

This module defines the request and response models for the Batches API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Literal

from pydantic import BaseModel, Field

from llama_stack_api.batches import BatchObject, ListBatchesResponse
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class CreateBatchRequest(BaseModel):
    """Request model for creating a batch."""

    input_file_id: str = Field(..., description="The ID of an uploaded file containing requests for the batch.")
    endpoint: str = Field(..., description="The endpoint to be used for all requests in the batch.")
    completion_window: Literal["24h"] = Field(
        ..., description="The time window within which the batch should be processed."
    )
    metadata: dict[str, str] | None = Field(default=None, description="Optional metadata for the batch.")
    idempotency_key: str | None = Field(
        default=None, description="Optional idempotency key. When provided, enables idempotent behavior."
    )


# Re-export response models for convenience
__all__ = ["CreateBatchRequest", "BatchObject", "ListBatchesResponse"]
