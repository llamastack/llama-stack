# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from pathlib import Path

import pytest
from openai.types.conversations.conversation import Conversation as OpenAIConversation
from openai.types.conversations.conversation_item import ConversationItem as OpenAIConversationItem
from pydantic import TypeAdapter

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
)
from llama_stack.core.conversations.conversations import (
    ConversationServiceConfig,
    ConversationServiceImpl,
)
from llama_stack.core.datatypes import StackRunConfig
from llama_stack.core.storage.datatypes import (
    ServerStoresConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from llama_stack.providers.utils.sqlstore.sqlstore import register_sqlstore_backends


@pytest.fixture
async def service():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conversations.db"

        storage = StorageConfig(
            backends={
                "sql_test": SqliteSqlStoreConfig(db_path=str(db_path)),
            },
            stores=ServerStoresConfig(
                conversations=SqlStoreReference(backend="sql_test", table_name="openai_conversations"),
            ),
        )
        register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})
        run_config = StackRunConfig(image_name="test", apis=[], providers={}, storage=storage)

        config = ConversationServiceConfig(run_config=run_config, policy=[])
        service = ConversationServiceImpl(config, {})
        await service.initialize()
        yield service


async def test_conversation_lifecycle(service):
    conversation = await service.create_conversation(metadata={"test": "data"})

    assert conversation.id.startswith("conv_")
    assert conversation.metadata == {"test": "data"}

    retrieved = await service.get_conversation(conversation.id)
    assert retrieved.id == conversation.id

    deleted = await service.openai_delete_conversation(conversation.id)
    assert deleted.id == conversation.id


async def test_conversation_items(service):
    conversation = await service.create_conversation()

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_test123",
            status="completed",
        )
    ]
    item_list = await service.add_items(conversation.id, items)

    assert len(item_list.data) == 1
    assert item_list.data[0].id == "msg_test123"

    items = await service.list_items(conversation.id)
    assert len(items.data) == 1


async def test_invalid_conversation_id(service):
    with pytest.raises(ValueError, match="Expected an ID that begins with 'conv_'"):
        await service._get_validated_conversation("invalid_id")


async def test_empty_parameter_validation(service):
    with pytest.raises(ValueError, match="Expected a non-empty value"):
        await service.retrieve("", "item_123")


async def test_openai_type_compatibility(service):
    conversation = await service.create_conversation(metadata={"test": "value"})

    conversation_dict = conversation.model_dump()
    openai_conversation = OpenAIConversation.model_validate(conversation_dict)

    for attr in ["id", "object", "created_at", "metadata"]:
        assert getattr(openai_conversation, attr) == getattr(conversation, attr)

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_test456",
            status="completed",
        )
    ]
    item_list = await service.add_items(conversation.id, items)

    for attr in ["object", "data", "first_id", "last_id", "has_more"]:
        assert hasattr(item_list, attr)
    assert item_list.object == "list"

    items = await service.list_items(conversation.id)
    item = await service.retrieve(conversation.id, items.data[0].id)
    item_dict = item.model_dump()

    openai_item_adapter = TypeAdapter(OpenAIConversationItem)
    openai_item_adapter.validate_python(item_dict)


async def test_policy_configuration():
    from llama_stack.core.access_control.datatypes import Action, Scope
    from llama_stack.core.datatypes import AccessRule

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conversations_policy.db"

        restrictive_policy = [
            AccessRule(forbid=Scope(principal="test_user", actions=[Action.CREATE, Action.READ], resource="*"))
        ]

        storage = StorageConfig(
            backends={
                "sql_test": SqliteSqlStoreConfig(db_path=str(db_path)),
            },
            stores=ServerStoresConfig(
                conversations=SqlStoreReference(backend="sql_test", table_name="openai_conversations"),
            ),
        )
        register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})
        run_config = StackRunConfig(image_name="test", apis=[], providers={}, storage=storage)

        config = ConversationServiceConfig(run_config=run_config, policy=restrictive_policy)
        service = ConversationServiceImpl(config, {})
        await service.initialize()

        assert service.policy == restrictive_policy
        assert len(service.policy) == 1
        assert service.policy[0].forbid is not None
