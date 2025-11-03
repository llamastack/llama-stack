# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import (
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemDeletedResource,
    ConversationItemInclude,
    ConversationItemList,
    Metadata,
)


@runtime_checkable
@trace_protocol
class ConversationService(Protocol):
    """Conversations

    Protocol for conversation management operations."""

    async def create_conversation(
        self, items: list[ConversationItem] | None = None, metadata: Metadata | None = None
    ) -> Conversation:
        """Create a conversation."""
        ...

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve a conversation."""
        ...

    async def update_conversation(self, conversation_id: str, metadata: Metadata) -> Conversation:
        """Update a conversation."""
        ...

    async def openai_delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        """Delete a conversation."""
        ...

    async def add_items(self, conversation_id: str, items: list[ConversationItem]) -> ConversationItemList:
        """Create items."""
        ...

    async def retrieve(self, conversation_id: str, item_id: str) -> ConversationItem:
        """Retrieve an item."""
        ...

    async def list_items(
        self,
        conversation_id: str,
        after: str | None = None,
        include: list[ConversationItemInclude] | None = None,
        limit: int | None = None,
        order: Literal["asc", "desc"] | None = None,
    ) -> ConversationItemList:
        """List items."""
        ...

    async def openai_delete_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> ConversationItemDeletedResource:
        """Delete an item."""
        ...
