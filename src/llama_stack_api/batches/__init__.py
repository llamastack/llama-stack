# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Batches API protocol and models.

This module contains the Batches protocol definition.
Pydantic models are defined in llama_stack_api.batches.models.
The FastAPI router is defined in llama_stack_api.batches.routes.
"""

from typing import Literal, Protocol, runtime_checkable

try:
    from openai.types import Batch as BatchObject
except ImportError as e:
    raise ImportError("OpenAI package is required for batches API. Please install it with: pip install openai") from e

# Import models for re-export
from llama_stack_api.batches.models import ListBatchesResponse


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
