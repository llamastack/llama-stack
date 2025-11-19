# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Batches API protocol and models.

This module contains the Batches protocol definition and related models.
The router implementation is in llama_stack.core.server.routers.batches.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack_api.schema_utils import json_schema_type

try:
    from openai.types import Batch as BatchObject
except ImportError as e:
    raise ImportError("OpenAI package is required for batches API. Please install it with: pip install openai") from e


@json_schema_type
class ListBatchesResponse(BaseModel):
    """Response containing a list of batch objects."""

    object: Literal["list"] = "list"
    data: list[BatchObject] = Field(..., description="List of batch objects")
    first_id: str | None = Field(default=None, description="ID of the first batch in the list")
    last_id: str | None = Field(default=None, description="ID of the last batch in the list")
    has_more: bool = Field(default=False, description="Whether there are more batches available")


@runtime_checkable
class Batches(Protocol):
    """
    The Batches API enables efficient processing of multiple requests in a single operation,
    particularly useful for processing large datasets, batch evaluation workflows, and
    cost-effective inference at scale.

    The API is designed to allow use of openai client libraries for seamless integration.

    This API provides the following extensions:
     - idempotent batch creation

    Note: This API is currently under active development and may undergo changes.
    """

    async def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: Literal["24h"],
        metadata: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> BatchObject: ...

    async def retrieve_batch(self, batch_id: str) -> BatchObject: ...

    async def cancel_batch(self, batch_id: str) -> BatchObject: ...

    async def list_batches(
        self,
        after: str | None = None,
        limit: int = 20,
    ) -> ListBatchesResponse: ...


__all__ = ["Batches", "BatchObject", "ListBatchesResponse"]
