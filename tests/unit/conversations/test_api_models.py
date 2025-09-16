# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.conversations.conversations import (
    Conversation,
    ConversationCreateRequest,
    ConversationItemList,
    ConversationMessage,
)


def test_conversation_create_request_defaults():
    request = ConversationCreateRequest()
    assert request.items == []
    assert request.metadata is None


def test_conversation_model_defaults():
    conversation = Conversation(
        id="conv_123456789",
        created_at=1234567890,
        metadata=None,
        object="conversation",
    )
    assert conversation.id == "conv_123456789"
    assert conversation.object == "conversation"
    assert conversation.metadata is None


def test_openai_client_compatibility():
    from openai.types.conversations.message import Message

    our_message = ConversationMessage(
        id="msg_123",
        content=[{"type": "input_text", "text": "Hello"}],
        role="user",
        status="in_progress",
        type="message",
    )

    openai_message = Message.model_validate(our_message.model_dump())
    assert openai_message.id == "msg_123"


def test_conversation_item_list():
    item_list = ConversationItemList(data=[])
    assert item_list.object == "list"
    assert item_list.data == []
    assert item_list.first_id is None
    assert item_list.last_id is None
    assert item_list.has_more is False
