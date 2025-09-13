# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.conversations.conversations import (
    Conversation,
    ConversationCreateRequest,
    ConversationItem,
    ConversationItemList,
)


def test_conversation_create_request_defaults():
    request = ConversationCreateRequest()
    assert request.items == []
    assert request.metadata is None


def test_conversation_model_defaults():
    conversation = Conversation(id="conv_123456789", created_at=1234567890)
    assert conversation.id == "conv_123456789"
    assert conversation.object == "conversation"
    assert conversation.items == []
    assert conversation.metadata is None


def test_conversation_item_model():
    mock_content = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
    item = ConversationItem(
        id="item_123456789", conversation_id="conv_123456789", created_at=1234567890, content=mock_content
    )
    assert item.id == "item_123456789"
    assert item.object == "conversation.item"
    assert item.conversation_id == "conv_123456789"


def test_conversation_item_list():
    item_list = ConversationItemList(data=[])
    assert item_list.object == "list"
    assert item_list.data == []
    assert item_list.first_id is None
    assert item_list.last_id is None
    assert item_list.has_more is False
