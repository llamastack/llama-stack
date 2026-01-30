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

from llama_stack.core.conversations.conversations import (
    ConversationServiceConfig,
    ConversationServiceImpl,
)
from llama_stack.core.datatypes import StackConfig
from llama_stack.core.storage.datatypes import (
    ServerStoresConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from llama_stack.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from llama_stack_api import OpenAIResponseInputMessageContentText, OpenAIResponseMessage
from llama_stack_api.conversations import (
    AddItemsRequest,
    CreateConversationRequest,
    DeleteConversationRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
)


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
                metadata=None,
                inference=None,
                prompts=None,
            ),
        )
        register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})
        stack_config = StackConfig(distro_name="test", apis=[], providers={}, storage=storage)

        config = ConversationServiceConfig(config=stack_config, policy=[])
        service = ConversationServiceImpl(config, {})
        await service.initialize()
        yield service


async def test_conversation_lifecycle(service):
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "data"}))

    assert conversation.id.startswith("conv_")
    assert conversation.metadata == {"test": "data"}

    retrieved = await service.get_conversation(GetConversationRequest(conversation_id=conversation.id))
    assert retrieved.id == conversation.id

    deleted = await service.openai_delete_conversation(DeleteConversationRequest(conversation_id=conversation.id))
    assert deleted.id == conversation.id


async def test_conversation_items(service):
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_test123",
            status="completed",
        )
    ]
    item_list = await service.add_items(conversation.id, AddItemsRequest(items=items))

    assert len(item_list.data) == 1
    assert item_list.data[0].id == "msg_test123"

    items_result = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    assert len(items_result.data) == 1


async def test_invalid_conversation_id(service):
    with pytest.raises(ValueError, match="Expected an ID that begins with 'conv_'"):
        await service._get_validated_conversation("invalid_id")


async def test_empty_parameter_validation(service):
    with pytest.raises(ValueError, match="Expected a non-empty value"):
        await service.retrieve(RetrieveItemRequest(conversation_id="", item_id="item_123"))


async def test_openai_type_compatibility(service):
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "value"}))

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
    item_list = await service.add_items(conversation.id, AddItemsRequest(items=items))

    for attr in ["object", "data", "first_id", "last_id", "has_more"]:
        assert hasattr(item_list, attr)
    assert item_list.object == "list"

    items_result = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    item = await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id=items_result.data[0].id))
    item_dict = item.model_dump()

    openai_item_adapter = TypeAdapter(OpenAIConversationItem)
    openai_item_adapter.validate_python(item_dict)


async def test_items_not_returned_on_creation_or_retrieval(service):
    """Test that items field is not returned when creating or retrieving a conversation.

    The items field is deprecated and kept for backward compatibility.
    Items should be accessed via the conversation_items table using the /items endpoint.
    """
    # Create a conversation
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "value"}))

    # Verify items field doesn't exist in serialized response
    conversation_dict = conversation.model_dump(exclude_none=True)
    assert "items" not in conversation_dict, "items should not be in creation response"

    # Retrieve the conversation
    retrieved = await service.get_conversation(GetConversationRequest(conversation_id=conversation.id))

    # Verify items field doesn't exist in retrieval response
    retrieved_dict = retrieved.model_dump(exclude_none=True)
    assert "items" not in retrieved_dict, "items should not be in retrieval response"


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
                metadata=None,
                inference=None,
                prompts=None,
            ),
        )
        register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})
        stack_config = StackConfig(distro_name="test", apis=[], providers={}, storage=storage)

        config = ConversationServiceConfig(config=stack_config, policy=restrictive_policy)
        service = ConversationServiceImpl(config, {})
        await service.initialize()

        assert service.policy == restrictive_policy
        assert len(service.policy) == 1
        assert service.policy[0].forbid is not None
