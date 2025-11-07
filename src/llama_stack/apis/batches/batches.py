# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, webmethod

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

    @webmethod(route="/batches", method="POST", level=LLAMA_STACK_API_V1)
    async def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: Literal["24h"],
        metadata: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> BatchObject:
        """Create a new batch for processing multiple API requests.

        :param input_file_id: The ID of an uploaded file containing requests for the batch.
        :param endpoint: The endpoint to be used for all requests in the batch.
        :param completion_window: The time window within which the batch should be processed.
        :param metadata: Optional metadata for the batch.
        :param idempotency_key: Optional idempotency key. When provided, enables idempotent behavior.
        :returns: The created batch object.
        """
        ...

    @webmethod(route="/batches/{batch_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def retrieve_batch(self, batch_id: str) -> BatchObject:
        """Retrieve information about a specific batch.

        :param batch_id: The ID of the batch to retrieve.
        :returns: The batch object.
        """
        ...

    @webmethod(route="/batches/{batch_id}/cancel", method="POST", level=LLAMA_STACK_API_V1)
    async def cancel_batch(self, batch_id: str) -> BatchObject:
        """Cancel a batch that is in progress.

        :param batch_id: The ID of the batch to cancel.
        :returns: The updated batch object.
        """
        ...

    @webmethod(route="/batches", method="GET", level=LLAMA_STACK_API_V1)
    async def list_batches(
        self,
        after: str | None = None,
        limit: int = 20,
    ) -> ListBatchesResponse:
        """List all batches for the current user.

        :param after: A cursor for pagination; returns batches after this batch ID.
        :param limit: Number of batches to return (default 20, max 100).
        :returns: A list of batch objects.
        """
        ...
