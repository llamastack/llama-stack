# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import secrets
import time
from typing import Any

from openai import NOT_GIVEN
from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.conversations.conversations import (
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemDeletedResource,
    ConversationItemList,
    Conversations,
    Metadata,
)
from llama_stack.core.datatypes import AccessRule
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.log import get_logger
from llama_stack.providers.utils.sqlstore.api import ColumnDefinition, ColumnType
from llama_stack.providers.utils.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.providers.utils.sqlstore.sqlstore import (
    SqliteSqlStoreConfig,
    SqlStoreConfig,
    sqlstore_impl,
)

logger = get_logger(name=__name__, category="openai::conversations")


class ConversationServiceConfig(BaseModel):
    """Configuration for the built-in conversation service.

    :param conversations_store: SQL store configuration for conversations (defaults to SQLite)
    :param policy: Access control rules
    """

    conversations_store: SqlStoreConfig = SqliteSqlStoreConfig(
        db_path=(DISTRIBS_BASE_DIR / "conversations.db").as_posix()
    )
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

        base_sql_store = sqlstore_impl(config.conversations_store)
        self.sql_store = AuthorizedSqlStore(base_sql_store, self.policy)

    async def initialize(self) -> None:
        """Initialize the store and create tables."""
        if isinstance(self.config.conversations_store, SqliteSqlStoreConfig):
            os.makedirs(os.path.dirname(self.config.conversations_store.db_path), exist_ok=True)

        await self.sql_store.create_table(
            "openai_conversations",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "items": ColumnType.JSON,
                "metadata": ColumnType.JSON,
            },
        )

    async def create_conversation(
        self, items: list[ConversationItem] | None = None, metadata: Metadata | None = None
    ) -> Conversation:
        """Create a conversation."""
        random_bytes = secrets.token_bytes(24)
        conversation_id = f"conv_{random_bytes.hex()}"
        created_at = int(time.time())

        items_json = []
        for item in items or []:
            items_json.append(item.model_dump())

        record_data = {
            "id": conversation_id,
            "created_at": created_at,
            "items": items_json,
            "metadata": metadata,
        }

        await self.sql_store.insert(
            table="openai_conversations",
            data=record_data,
        )

        conversation = Conversation(
            id=conversation_id,
            created_at=created_at,
            metadata=metadata,
            object="conversation",
        )

        logger.info(f"Created conversation {conversation_id}")
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation with the given ID."""
        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": conversation_id})

        if record is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        return Conversation(
            id=record["id"], created_at=record["created_at"], metadata=record.get("metadata"), object="conversation"
        )

    async def update_conversation(self, conversation_id: str, metadata: Metadata) -> Conversation:
        """Update a conversation's metadata with the given ID"""
        await self.sql_store.update(
            table="openai_conversations", data={"metadata": metadata}, where={"id": conversation_id}
        )

        return await self.get_conversation(conversation_id)

    async def openai_delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        """Delete a conversation with the given ID."""
        await self.sql_store.delete(table="openai_conversations", where={"id": conversation_id})

        logger.info(f"Deleted conversation {conversation_id}")
        return ConversationDeletedResource(id=conversation_id)

    def _validate_conversation_id(self, conversation_id: str) -> None:
        """Validate conversation ID format."""
        if not conversation_id.startswith("conv_"):
            raise ValueError(
                f"Invalid 'conversation_id': '{conversation_id}'. Expected an ID that begins with 'conv_'."
            )

    async def _get_validated_conversation(self, conversation_id: str) -> Conversation:
        """Validate conversation ID and return the conversation if it exists."""
        self._validate_conversation_id(conversation_id)
        return await self.get_conversation(conversation_id)

    async def create(self, conversation_id: str, items: list[ConversationItem]) -> ConversationItemList:
        """Create items in the conversation."""
        await self._get_validated_conversation(conversation_id)

        created_items = []

        for item in items:
            # Generate item ID based on item type
            random_bytes = secrets.token_bytes(24)
            if item.type == "message":
                item_id = f"msg_{random_bytes.hex()}"
            else:
                item_id = f"item_{random_bytes.hex()}"

            # Create a copy of the item with the generated ID and completed status
            item_dict = item.model_dump()
            item_dict["id"] = item_id
            if "status" not in item_dict:
                item_dict["status"] = "completed"

            created_items.append(item_dict)

        # Get existing items from database
        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": conversation_id})
        existing_items = record.get("items", []) if record else []

        updated_items = existing_items + created_items
        await self.sql_store.update(
            table="openai_conversations", data={"items": updated_items}, where={"id": conversation_id}
        )

        logger.info(f"Created {len(created_items)} items in conversation {conversation_id}")

        # Convert created items (dicts) to proper ConversationItem types
        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [adapter.validate_python(item_dict) for item_dict in created_items]

        return ConversationItemList(
            data=response_items,
            first_id=created_items[0]["id"] if created_items else None,
            last_id=created_items[-1]["id"] if created_items else None,
            has_more=False,
        )

    async def retrieve(self, conversation_id: str, item_id: str) -> ConversationItem:
        """Retrieve a conversation item."""
        if not conversation_id:
            raise ValueError(f"Expected a non-empty value for `conversation_id` but received {conversation_id!r}")
        if not item_id:
            raise ValueError(f"Expected a non-empty value for `item_id` but received {item_id!r}")

        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": conversation_id})
        items = record.get("items", []) if record else []

        for item in items:
            if isinstance(item, dict) and item.get("id") == item_id:
                adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
                return adapter.validate_python(item)

        raise ValueError(f"Item {item_id} not found in conversation {conversation_id}")

    async def list(self, conversation_id: str, after=NOT_GIVEN, include=NOT_GIVEN, limit=NOT_GIVEN, order=NOT_GIVEN):
        """List items in the conversation."""
        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": conversation_id})
        items = record.get("items", []) if record else []

        if order != NOT_GIVEN and order == "asc":
            items = items
        else:
            items = list(reversed(items))

        actual_limit = 20
        if limit != NOT_GIVEN and isinstance(limit, int):
            actual_limit = limit

        items = items[:actual_limit]

        # Items from database are stored as dicts, convert them to ConversationItem
        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [
            adapter.validate_python(item) if isinstance(item, dict) else item for item in items
        ]

        # Get first and last IDs from converted response items
        first_id = response_items[0].id if response_items else None
        last_id = response_items[-1].id if response_items else None

        return ConversationItemList(
            data=response_items,
            first_id=first_id,
            last_id=last_id,
            has_more=False,
        )

    async def openai_delete_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> ConversationItemDeletedResource:
        """Delete a conversation item."""
        if not conversation_id:
            raise ValueError(f"Expected a non-empty value for `conversation_id` but received {conversation_id!r}")
        if not item_id:
            raise ValueError(f"Expected a non-empty value for `item_id` but received {item_id!r}")

        _ = await self._get_validated_conversation(conversation_id)  # executes validation

        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": conversation_id})
        items = record.get("items", []) if record else []

        updated_items = []
        item_found = False

        for item in items:
            current_item_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if current_item_id != item_id:
                updated_items.append(item)
            else:
                item_found = True

        if not item_found:
            raise ValueError(f"Item {item_id} not found in conversation {conversation_id}")

        await self.sql_store.update(
            table="openai_conversations", data={"items": updated_items}, where={"id": conversation_id}
        )

        logger.info(f"Deleted item {item_id} from conversation {conversation_id}")
        return ConversationItemDeletedResource(id=item_id)
