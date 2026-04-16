# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .models import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
    CreateMessageBatchRequest,
    ListMessageBatchesResponse,
    MessageBatch,
    MessageBatchIndividualResponse,
)


@runtime_checkable
class Messages(Protocol):
    """Protocol for the Anthropic Messages API."""

    async def create_message(
        self,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]: ...

    async def count_message_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse: ...

    async def create_message_batch(
        self,
        request: CreateMessageBatchRequest,
    ) -> MessageBatch: ...

    async def retrieve_message_batch(
        self,
        batch_id: str,
    ) -> MessageBatch: ...

    async def list_message_batches(
        self,
        limit: int = 20,
        before_id: str | None = None,
        after_id: str | None = None,
    ) -> ListMessageBatchesResponse: ...

    async def cancel_message_batch(
        self,
        batch_id: str,
    ) -> MessageBatch: ...

    async def retrieve_message_batch_results(
        self,
        batch_id: str,
    ) -> AsyncIterator[MessageBatchIndividualResponse]: ...
