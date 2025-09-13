# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from pathlib import Path

import pytest

from llama_stack.apis.conversations.conversations import (
    ConversationCreateRequest,
    ConversationItemCreateRequest,
)
from llama_stack.core.conversations.conversations import (
    ConversationServiceConfig,
    ConversationServiceImpl,
)
from llama_stack.core.datatypes import ConversationsStoreConfig, StackRunConfig
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig


@pytest.fixture
async def service():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conversations.db"

        config = ConversationServiceConfig(
            run_config=StackRunConfig(
                image_name="test",
                providers={},
                conversations_store=ConversationsStoreConfig(
                    sql_store_config=SqliteSqlStoreConfig(db_path=str(db_path))
                ),
            )
        )
        service = ConversationServiceImpl(config, {})
        await service.initialize()
        yield service


async def test_conversation_lifecycle(service):
    request = ConversationCreateRequest(metadata={"test": "data"})
    conversation = await service.create_conversation(request)

    assert conversation.id.startswith("conv_")
    assert conversation.metadata == {"test": "data"}

    retrieved = await service.get_conversation(conversation.id)
    assert retrieved.id == conversation.id

    deleted = await service.delete_conversation(conversation.id)
    assert deleted.id == conversation.id


async def test_conversation_items(service):
    conversation = await service.create_conversation(ConversationCreateRequest())

    items_request = ConversationItemCreateRequest(
        items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
    )
    item_list = await service.create(conversation.id, items_request)

    assert len(item_list.data) == 1
    assert item_list.data[0].id.startswith("msg_")

    items = await service.list(conversation.id)
    assert len(items.data) == 1


async def test_invalid_conversation_id(service):
    with pytest.raises(ValueError, match="Expected an ID that begins with 'conv_'"):
        await service._get_validated_conversation("invalid_id")


async def test_empty_parameter_validation(service):
    with pytest.raises(ValueError, match="Expected a non-empty value"):
        await service.retrieve("", "item_123")


async def test_openai_compatibility(service):
    request = ConversationCreateRequest(metadata={"key": "value"})
    conversation = await service.create_conversation(request)

    for attr in ["id", "object", "created_at", "metadata"]:
        assert hasattr(conversation, attr)
    assert conversation.object == "conversation"

    items_request = ConversationItemCreateRequest(
        items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Test"}]}]
    )
    item_list = await service.create(conversation.id, items_request)

    for attr in ["object", "data", "first_id", "last_id", "has_more"]:
        assert hasattr(item_list, attr)
    assert item_list.object == "list"

    message = item_list.data[0]
    for attr in ["type", "role", "status", "id", "content"]:
        assert hasattr(message, attr)
    assert message.type == "message"
    assert message.id.startswith("msg_")
