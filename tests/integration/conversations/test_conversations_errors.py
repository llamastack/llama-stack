# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Error handling tests for the Llama Stack Conversations API.

These tests verify that errors emitted by Llama Stack are correctly typed
and handled by the OpenAI Python SDK, ensuring users don't have breaking
experiences when error conditions occur.

The OpenAI SDK expects specific HTTP status codes to trigger specific
exception types:
    - 400 -> openai.BadRequestError
    - 401 -> openai.AuthenticationError
    - 404 -> openai.NotFoundError
    - 409 -> openai.ConflictError
    - 422 -> openai.UnprocessableEntityError
    - 429 -> openai.RateLimitError
    - 5xx -> openai.InternalServerError

See: https://github.com/openai/openai-python/blob/main/src/openai/_exceptions.py
"""

import pytest
from openai import BadRequestError, NotFoundError


class TestConversationNotFoundErrors:
    """Tests for 404 Not Found errors when conversations don't exist.

    These tests verify that attempting to access nonexistent conversations
    returns 404 and triggers openai.NotFoundError in the SDK.
    """

    def test_retrieve_nonexistent_conversation_raises_not_found(self, openai_client):
        """Test that retrieving a nonexistent conversation returns 404."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.retrieve("conv_nonexistent123456789012")

        assert exc_info.value.status_code == 404
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "conversation" in error_msg

    def test_update_nonexistent_conversation_raises_not_found(self, openai_client):
        """Test that updating a nonexistent conversation returns 404."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.update(
                "conv_nonexistent123456789012",
                metadata={"topic": "test"},
            )

        assert exc_info.value.status_code == 404

    def test_delete_nonexistent_conversation_raises_not_found(self, openai_client):
        """Test that deleting a nonexistent conversation returns 404."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.delete("conv_nonexistent123456789012")

        assert exc_info.value.status_code == 404


class TestConversationItemNotFoundErrors:
    """Tests for 404 Not Found errors when conversation items don't exist.

    These tests verify that attempting to access nonexistent items
    returns 404 and triggers openai.NotFoundError in the SDK.
    """

    def test_retrieve_nonexistent_item_raises_not_found(self, openai_client):
        """Test that retrieving a nonexistent item returns 404."""
        conversation = openai_client.conversations.create()

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.retrieve(
                "item_nonexistent123456789012",
                conversation_id=conversation.id,
            )

        assert exc_info.value.status_code == 404
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "item" in error_msg

    def test_delete_nonexistent_item_raises_not_found(self, openai_client):
        """Test that deleting a nonexistent item returns 404."""
        conversation = openai_client.conversations.create()

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.delete(
                "item_nonexistent123456789012",
                conversation_id=conversation.id,
            )

        assert exc_info.value.status_code == 404

    def test_retrieve_item_nonexistent_conversation_raises_not_found(self, openai_client):
        """Test that retrieving item from nonexistent conversation returns 404."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.retrieve(
                "item_123456789012",
                conversation_id="conv_nonexistent123456789012",
            )

        assert exc_info.value.status_code == 404

    def test_delete_item_nonexistent_conversation_raises_not_found(self, openai_client):
        """Test that deleting item from nonexistent conversation returns 404."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.delete(
                "item_123456789012",
                conversation_id="conv_nonexistent123456789012",
            )

        assert exc_info.value.status_code == 404


class TestInvalidConversationIdErrors:
    """Tests for 400 Bad Request errors when conversation IDs are invalid.

    These tests verify that invalid conversation ID formats return 400
    and trigger openai.BadRequestError in the SDK.
    """

    def test_retrieve_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that retrieving with invalid conversation ID format returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.retrieve("invalid-no-conv-prefix")

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "conv" in error_msg or "invalid" in error_msg

    def test_update_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that updating with invalid conversation ID format returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.update(
                "invalid-no-conv-prefix",
                metadata={"topic": "test"},
            )

        assert exc_info.value.status_code == 400

    def test_delete_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that deleting with invalid conversation ID format returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.delete("invalid-no-conv-prefix")

        assert exc_info.value.status_code == 400

    def test_items_list_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that listing items with invalid conversation ID returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.items.list("invalid-no-conv-prefix")

        assert exc_info.value.status_code == 400

    def test_items_create_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that creating items with invalid conversation ID returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.items.create(
                "invalid-no-conv-prefix",
                items=[{"type": "message", "role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 400

    def test_items_retrieve_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that retrieving item with invalid conversation ID returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.items.retrieve(
                "item_123456789012",
                conversation_id="invalid-no-conv-prefix",
            )

        assert exc_info.value.status_code == 400

    def test_items_delete_invalid_conversation_id_raises_bad_request(self, openai_client):
        """Test that deleting item with invalid conversation ID returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.items.delete(
                "item_123456789012",
                conversation_id="invalid-no-conv-prefix",
            )

        assert exc_info.value.status_code == 400


class TestDeletedConversationErrors:
    """Tests for errors when operating on deleted conversations.

    These tests verify that operations on deleted conversations return
    appropriate errors.
    """

    def test_retrieve_deleted_conversation_raises_not_found(self, openai_client):
        """Test that retrieving a deleted conversation returns 404."""
        conversation = openai_client.conversations.create()
        openai_client.conversations.delete(conversation.id)

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.retrieve(conversation.id)

        assert exc_info.value.status_code == 404

    def test_update_deleted_conversation_raises_not_found(self, openai_client):
        """Test that updating a deleted conversation returns 404."""
        conversation = openai_client.conversations.create()
        openai_client.conversations.delete(conversation.id)

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.update(conversation.id, metadata={"topic": "test"})

        assert exc_info.value.status_code == 404

    def test_list_items_deleted_conversation_raises_not_found(self, openai_client):
        """Test that listing items of a deleted conversation returns 404."""
        conversation = openai_client.conversations.create()
        openai_client.conversations.delete(conversation.id)

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.list(conversation.id)

        assert exc_info.value.status_code == 404

    def test_create_items_deleted_conversation_raises_not_found(self, openai_client):
        """Test that creating items in a deleted conversation returns 404."""
        conversation = openai_client.conversations.create()
        openai_client.conversations.delete(conversation.id)

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.create(
                conversation.id,
                items=[{"type": "message", "role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 404


class TestConversationViaResponsesAPIErrors:
    """Tests for conversation errors accessed through the Responses API.

    These tests verify SDK compatibility for conversation-related operations
    accessed through the `conversation` parameter in responses.create().
    """

    def test_invalid_conversation_id_format_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that an invalid conversation ID format returns 400 and triggers
        openai.BadRequestError in the SDK.
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                conversation="invalid-format-no-conv-prefix",
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "conv" in error_msg or "invalid" in error_msg

    def test_nonexistent_conversation_raises_not_found_error(self, openai_client, text_model_id):
        """
        Test that referencing a nonexistent conversation returns 404 and triggers
        openai.NotFoundError in the SDK.
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                conversation="conv_nonexistent123456",
            )

        assert exc_info.value.status_code == 404
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "conversation" in error_msg
