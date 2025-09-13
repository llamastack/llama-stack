# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import secrets
import time
from typing import Any

from openai import NOT_GIVEN
from pydantic import BaseModel

from llama_stack.apis.conversations.conversations import (
    Conversation,
    ConversationCreateRequest,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemCreateRequest,
    ConversationItemDeletedResource,
    ConversationItemList,
    Conversations,
    ConversationUpdateRequest,
)
from llama_stack.core.datatypes import AccessRule, StackRunConfig
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.log import get_logger
from llama_stack.providers.utils.sqlstore.api import ColumnDefinition, ColumnType
from llama_stack.providers.utils.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.providers.utils.sqlstore.sqlstore import PostgresSqlStoreConfig, SqliteSqlStoreConfig, sqlstore_impl

logger = get_logger(name=__name__, category="openai::conversations")


class ConversationServiceConfig(BaseModel):
    """Configuration for the built-in conversation service.

    :param run_config: Stack run configuration containing distribution info
    """

    run_config: StackRunConfig


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
        self.policy: list[AccessRule] = []

        conversations_store_config = config.run_config.conversations_store
        if conversations_store_config is None:
            sql_store_config: SqliteSqlStoreConfig | PostgresSqlStoreConfig = SqliteSqlStoreConfig(
                db_path=(DISTRIBS_BASE_DIR / config.run_config.image_name / "conversations.db").as_posix()
            )
        elif hasattr(conversations_store_config, "sql_store_config"):
            sql_store_config = conversations_store_config.sql_store_config
        else:
            # conversations_store_config is already a SqlStore config
            sql_store_config = conversations_store_config

        base_sql_store = sqlstore_impl(sql_store_config)
        self.sql_store = AuthorizedSqlStore(base_sql_store)

    async def initialize(self) -> None:
        """Initialize the store and create tables."""
        await self.sql_store.create_table(
            "openai_conversations",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "items": ColumnType.JSON,
                "metadata": ColumnType.JSON,
            },
        )

    async def create_conversation(self, request: ConversationCreateRequest) -> Conversation:
        """Create a conversation."""
        random_bytes = secrets.token_bytes(24)
        conversation_id = f"conv_{random_bytes.hex()}"
        created_at = int(time.time())

        items_json = []
        for item in request.items or []:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else item
            items_json.append(item_dict)

        record_data = {
            "id": conversation_id,
            "created_at": created_at,
            "items": items_json,
            "metadata": request.metadata,
        }

        await self.sql_store.insert(
            table="openai_conversations",
            data=record_data,
        )

        conversation = Conversation(
            id=conversation_id, created_at=created_at, items=request.items or [], metadata=request.metadata
        )

        logger.info(f"Created conversation {conversation_id}")
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation with the given ID."""
        record = await self.sql_store.fetch_one(
            table="openai_conversations", policy=self.policy, where={"id": conversation_id}
        )

        if record is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        items = []
        for item_data in record.get("items", []):
            items.append(item_data)

        return Conversation(
            id=record["id"], created_at=record["created_at"], items=items, metadata=record.get("metadata")
        )

    async def update_conversation(self, conversation_id: str, request: ConversationUpdateRequest) -> Conversation:
        """Update a conversation's metadata with the given ID"""
        await self.sql_store.update(
            table="openai_conversations", data={"metadata": request.metadata}, where={"id": conversation_id}
        )

        return await self.get_conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
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
        try:
            return await self.get_conversation(conversation_id)
        except ValueError:
            raise ValueError(
                f"Invalid 'conversation_id': '{conversation_id}'. Expected an ID that begins with 'conv'."
            ) from None

    async def create(self, conversation_id: str, request: ConversationItemCreateRequest) -> ConversationItemList:
        """Create items in the conversation."""
        conversation = await self._get_validated_conversation(conversation_id)

        created_items = []

        for item in request.items:
            # Generate item ID based on item type
            random_bytes = secrets.token_bytes(24)
            if hasattr(item, "type") and item.type == "message":
                item_id = f"msg_{random_bytes.hex()}"
            else:
                item_id = f"item_{random_bytes.hex()}"

            # Create a copy of the item with the generated ID and completed status
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            item_dict["id"] = item_id
            if "status" not in item_dict:
                item_dict["status"] = "completed"

            created_items.append(item_dict)

        conversation_items_json = []
        for item in conversation.items:
            if hasattr(item, "model_dump"):
                conversation_items_json.append(item.model_dump())
            elif isinstance(item, dict):
                conversation_items_json.append(item)
            else:
                # For any other type, convert to dict if possible
                conversation_items_json.append(dict(item) if hasattr(item, "__dict__") else {})

        updated_items = conversation_items_json + created_items
        await self.sql_store.update(
            table="openai_conversations", data={"items": updated_items}, where={"id": conversation_id}
        )

        logger.info(f"Created {len(created_items)} items in conversation {conversation_id}")
        # Return the original items from the request, which are already OpenAIResponseInput types
        response_items = request.items

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

        conversation = await self._get_validated_conversation(conversation_id)

        for item in conversation.items:
            if isinstance(item, dict) and item.get("id") == item_id:
                return ConversationItem(
                    id=item_id, conversation_id=conversation_id, created_at=conversation.created_at, content=item
                )
            elif hasattr(item, "id") and item.id == item_id:
                return ConversationItem(
                    id=item_id, conversation_id=conversation_id, created_at=conversation.created_at, content=item
                )

        raise ValueError(f"Item {item_id} not found in conversation {conversation_id}")

    async def list(self, conversation_id: str, after=NOT_GIVEN, include=NOT_GIVEN, limit=NOT_GIVEN, order=NOT_GIVEN):
        """List items in the conversation."""
        conversation = await self._get_validated_conversation(conversation_id)

        items = conversation.items or []

        if order != NOT_GIVEN and order == "asc":
            items = items
        else:
            items = list(reversed(items))

        actual_limit = 20
        if limit != NOT_GIVEN and isinstance(limit, int):
            actual_limit = limit

        items = items[:actual_limit]

        response_items = []
        for item in items:
            # Items can be either dicts (stored in DB) or OpenAI objects (in memory)
            response_items.append(item)

        return ConversationItemList(
            data=response_items,
            first_id=items[0].get("id")
            if items and isinstance(items[0], dict)
            else (getattr(items[0], "id", None) if items else None),
            last_id=items[-1].get("id")
            if items and isinstance(items[-1], dict)
            else (getattr(items[-1], "id", None) if items else None),
            has_more=False,
        )

    async def delete(self, conversation_id: str, item_id: str) -> ConversationItemDeletedResource:
        """Delete a conversation item."""
        if not conversation_id:
            raise ValueError(f"Expected a non-empty value for `conversation_id` but received {conversation_id!r}")
        if not item_id:
            raise ValueError(f"Expected a non-empty value for `item_id` but received {item_id!r}")

        conversation = await self._get_validated_conversation(conversation_id)

        updated_items = []
        item_found = False

        for item in conversation.items:
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
