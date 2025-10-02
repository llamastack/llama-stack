# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest


@pytest.mark.integration
class TestOpenAIConversations:
    # TODO: Update to compat_client after client-SDK is generated
    def test_conversation_create(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "demo"}, items=[{"type": "message", "role": "user", "content": "Hello!"}]
        )

        assert conversation.id.startswith("conv_")
        assert conversation.object == "conversation"
        assert conversation.metadata["topic"] == "demo"
        assert isinstance(conversation.created_at, int)

    def test_conversation_retrieve(self, openai_client):
        conversation = openai_client.conversations.create(metadata={"topic": "demo"})

        retrieved = openai_client.conversations.retrieve(conversation.id)

        assert retrieved.id == conversation.id
        assert retrieved.object == "conversation"
        assert retrieved.metadata["topic"] == "demo"
        assert retrieved.created_at == conversation.created_at

    def test_conversation_update(self, openai_client):
        conversation = openai_client.conversations.create(metadata={"topic": "demo"})

        updated = openai_client.conversations.update(conversation.id, metadata={"topic": "project-x"})

        assert updated.id == conversation.id
        assert updated.metadata["topic"] == "project-x"
        assert updated.created_at == conversation.created_at

    def test_conversation_delete(self, openai_client):
        conversation = openai_client.conversations.create(metadata={"topic": "demo"})

        deleted = openai_client.conversations.delete(conversation.id)

        assert deleted.id == conversation.id
        assert deleted.object == "conversation.deleted"
        assert deleted.deleted is True

    def test_conversation_items_create(self, openai_client):
        conversation = openai_client.conversations.create()

        items = openai_client.conversations.items.create(
            conversation.id,
            items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "How are you?"}]},
            ],
        )

        assert items.object == "list"
        assert len(items.data) == 2
        assert items.data[0].content[0].text == "Hello!"
        assert items.data[1].content[0].text == "How are you?"
        assert items.first_id == items.data[0].id
        assert items.last_id == items.data[1].id
        assert items.has_more is False

    def test_conversation_items_list(self, openai_client):
        conversation = openai_client.conversations.create()

        openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
        )

        items = openai_client.conversations.items.list(conversation.id, limit=10)

        assert items.object == "list"
        assert len(items.data) >= 1
        assert items.data[0].type == "message"
        assert items.data[0].role == "user"
        assert hasattr(items, "first_id")
        assert hasattr(items, "last_id")
        assert hasattr(items, "has_more")

    def test_conversation_item_retrieve(self, openai_client):
        conversation = openai_client.conversations.create()

        created_items = openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
        )

        item_id = created_items.data[0].id
        item = openai_client.conversations.items.retrieve(item_id, conversation_id=conversation.id)

        assert item.id == item_id
        assert item.type == "message"
        assert item.role == "user"
        assert item.content[0].text == "Hello!"

    def test_conversation_item_delete(self, openai_client):
        conversation = openai_client.conversations.create()

        created_items = openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
        )

        item_id = created_items.data[0].id
        deleted = openai_client.conversations.items.delete(item_id, conversation_id=conversation.id)

        assert deleted.id == item_id
        assert deleted.object == "conversation.item.deleted"
        assert deleted.deleted is True

    def test_full_workflow(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "workflow-test"}, items=[{"type": "message", "role": "user", "content": "Hello!"}]
        )

        openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Follow up"}]}],
        )

        all_items = openai_client.conversations.items.list(conversation.id)
        assert len(all_items.data) >= 2

        updated = openai_client.conversations.update(conversation.id, metadata={"topic": "workflow-complete"})
        assert updated.metadata["topic"] == "workflow-complete"

        openai_client.conversations.delete(conversation.id)
