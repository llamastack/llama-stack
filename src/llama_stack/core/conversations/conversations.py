# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import secrets
import time
from typing import Any

from pydantic import BaseModel, TypeAdapter

from llama_stack.core.datatypes import AccessRule, StackConfig
from llama_stack.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.core.storage.sqlstore.sqlstore import sqlstore_impl
from llama_stack.log import get_logger
from llama_stack_api.conversations import (
    AddItemsRequest,
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemDeletedResource,
    ConversationItemList,
    Conversations,
    CreateConversationRequest,
    DeleteConversationRequest,
    DeleteItemRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
    UpdateConversationRequest,
)
from llama_stack_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="openai_conversations")


class ConversationServiceConfig(BaseModel):
    """Configuration for the built-in conversation service.

    :param run_config: Stack run configuration for resolving persistence
    :param policy: Access control rules
    """

    config: StackConfig
    policy: list[AccessRule] = []


async def get_provider_impl(config: ConversationServiceConfig, deps: dict[Any, Any]):
    """Get the conversation service implementation."""
    impl = ConversationServiceImpl(config, deps)
    await impl.initialize()
    return impl


class ConversationServiceImpl(Conversations):
    """Built-in conversation service implementation using AuthorizedSqlStore."""

    def __init__(self, config: ConversationServiceConfig, deps: dict[Any, Any]):
        self.config = config
        self.deps = deps
        self.policy = config.policy

        # Use conversations store reference from run config
        conversations_ref = config.config.storage.stores.conversations
        if not conversations_ref:
            raise ValueError("storage.stores.conversations must be configured in run config")

        base_sql_store = sqlstore_impl(conversations_ref)
        self.sql_store = AuthorizedSqlStore(base_sql_store, self.policy)

    async def initialize(self) -> None:
        """Initialize the store and create tables."""
        await self.sql_store.create_table(
            "openai_conversations",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "items": ColumnType.JSON,  # Deprecated: kept for backward compatibility, use conversation_items table instead
                "metadata": ColumnType.JSON,
            },
        )

        await self.sql_store.create_table(
            "conversation_items",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "conversation_id": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "item_data": ColumnType.JSON,
            },
        )

    async def create_conversation(self, request: CreateConversationRequest) -> Conversation:
        """Create a conversation."""
        random_bytes = secrets.token_bytes(24)
        conversation_id = f"conv_{random_bytes.hex()}"
        created_at = int(time.time())

        record_data = {
            "id": conversation_id,
            "created_at": created_at,
            "metadata": request.metadata,
        }

        await self.sql_store.insert(
            table="openai_conversations",
            data=record_data,
        )

        if request.items:
            item_records = []
            for item in request.items:
                item_dict = item.model_dump()
                item_id = self._get_or_generate_item_id(item, item_dict)

                item_record = {
                    "id": item_id,
                    "conversation_id": conversation_id,
                    "created_at": created_at,
                    "item_data": item_dict,
                }

                item_records.append(item_record)

            await self.sql_store.insert(table="conversation_items", data=item_records)

        conversation = Conversation(
            id=conversation_id,
            created_at=created_at,
            metadata=request.metadata,
            object="conversation",
        )

        logger.debug(f"Created conversation {conversation_id}")
        return conversation

    async def get_conversation(self, request: GetConversationRequest) -> Conversation:
        """Get a conversation with the given ID."""
        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": request.conversation_id})

        if record is None:
            raise ValueError(f"Conversation {request.conversation_id} not found")

        return Conversation(
            id=record["id"], created_at=record["created_at"], metadata=record.get("metadata"), object="conversation"
        )

    async def update_conversation(self, conversation_id: str, request: UpdateConversationRequest) -> Conversation:
        """Update a conversation's metadata with the given ID"""
        await self.sql_store.update(
            table="openai_conversations", data={"metadata": request.metadata}, where={"id": conversation_id}
        )

        return await self.get_conversation(GetConversationRequest(conversation_id=conversation_id))

    async def openai_delete_conversation(self, request: DeleteConversationRequest) -> ConversationDeletedResource:
        """Delete a conversation with the given ID."""
        await self.sql_store.delete(table="openai_conversations", where={"id": request.conversation_id})

        logger.debug(f"Deleted conversation {request.conversation_id}")
        return ConversationDeletedResource(id=request.conversation_id)

    def _validate_conversation_id(self, conversation_id: str) -> None:
        """Validate conversation ID format."""
        if not conversation_id.startswith("conv_"):
            raise ValueError(
                f"Invalid 'conversation_id': '{conversation_id}'. Expected an ID that begins with 'conv_'."
            )

    def _get_or_generate_item_id(self, item: ConversationItem, item_dict: dict) -> str:
        """Get existing item ID or generate one if missing."""
        if item.id is None:
            random_bytes = secrets.token_bytes(24)
            if item.type == "message":
                item_id = f"msg_{random_bytes.hex()}"
            else:
                item_id = f"item_{random_bytes.hex()}"
            item_dict["id"] = item_id
            return item_id
        return item.id

    async def _get_validated_conversation(self, conversation_id: str) -> Conversation:
        """Validate conversation ID and return the conversation if it exists."""
        self._validate_conversation_id(conversation_id)
        return await self.get_conversation(GetConversationRequest(conversation_id=conversation_id))

    async def add_items(self, conversation_id: str, request: AddItemsRequest) -> ConversationItemList:
        """Create (add) items to a conversation."""
        await self._get_validated_conversation(conversation_id)

        created_items = []
        base_time = int(time.time())

        for i, item in enumerate(request.items):
            item_dict = item.model_dump()
            item_id = self._get_or_generate_item_id(item, item_dict)

            # make each timestamp unique to maintain order
            created_at = base_time + i

            item_record = {
                "id": item_id,
                "conversation_id": conversation_id,
                "created_at": created_at,
                "item_data": item_dict,
            }

            await self.sql_store.upsert(
                table="conversation_items",
                data=item_record,
                conflict_columns=["id"],
            )

            created_items.append(item_dict)

        logger.debug(f"Created {len(created_items)} items in conversation {conversation_id}")

        # Convert created items (dicts) to proper ConversationItem types
        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [adapter.validate_python(item_dict) for item_dict in created_items]

        return ConversationItemList(
            data=response_items,
            first_id=created_items[0]["id"] if created_items else None,
            last_id=created_items[-1]["id"] if created_items else None,
            has_more=False,
        )

    async def retrieve(self, request: RetrieveItemRequest) -> ConversationItem:
        """Retrieve a conversation item."""
        if not request.conversation_id:
            raise ValueError(
                f"Expected a non-empty value for `conversation_id` but received {request.conversation_id!r}"
            )
        if not request.item_id:
            raise ValueError(f"Expected a non-empty value for `item_id` but received {request.item_id!r}")

        # Get item from conversation_items table
        record = await self.sql_store.fetch_one(
            table="conversation_items", where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        if record is None:
            raise ValueError(f"Item {request.item_id} not found in conversation {request.conversation_id}")

        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        return adapter.validate_python(record["item_data"])

    async def list_items(self, request: ListItemsRequest) -> ConversationItemList:
        """List items in the conversation."""
        if not request.conversation_id:
            raise ValueError(
                f"Expected a non-empty value for `conversation_id` but received {request.conversation_id!r}"
            )

        # check if conversation exists
        await self.get_conversation(GetConversationRequest(conversation_id=request.conversation_id))

        result = await self.sql_store.fetch_all(
            table="conversation_items", where={"conversation_id": request.conversation_id}
        )
        records = result.data

        if request.order is not None and request.order == "asc":
            records.sort(key=lambda x: x["created_at"])
        else:
            records.sort(key=lambda x: x["created_at"], reverse=True)

        actual_limit = request.limit or 20

        records = records[:actual_limit]
        items = [record["item_data"] for record in records]

        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [adapter.validate_python(item) for item in items]

        first_id = response_items[0].id if response_items else None
        last_id = response_items[-1].id if response_items else None

        return ConversationItemList(
            data=response_items,
            first_id=first_id,
            last_id=last_id,
            has_more=False,
        )

    async def openai_delete_conversation_item(self, request: DeleteItemRequest) -> ConversationItemDeletedResource:
        """Delete a conversation item."""
        if not request.conversation_id:
            raise ValueError(
                f"Expected a non-empty value for `conversation_id` but received {request.conversation_id!r}"
            )
        if not request.item_id:
            raise ValueError(f"Expected a non-empty value for `item_id` but received {request.item_id!r}")

        _ = await self._get_validated_conversation(request.conversation_id)

        record = await self.sql_store.fetch_one(
            table="conversation_items", where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        if record is None:
            raise ValueError(f"Item {request.item_id} not found in conversation {request.conversation_id}")

        await self.sql_store.delete(
            table="conversation_items", where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        logger.debug(f"Deleted item {request.item_id} from conversation {request.conversation_id}")
        return ConversationItemDeletedResource(id=request.item_id)

    async def shutdown(self) -> None:
        pass
