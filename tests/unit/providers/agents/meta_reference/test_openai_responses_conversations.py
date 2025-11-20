# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack_api.common.errors import (
    ConversationNotFoundError,
    InvalidConversationIdError,
)
from llama_stack_api.conversations import (
    ConversationItemList,
)
from llama_stack_api.openai_responses import (
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseOutputMessageContentOutputText,
)

# Import existing fixtures from the main responses test file
pytest_plugins = ["tests.unit.providers.agents.meta_reference.test_openai_responses"]

from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)


@pytest.fixture
def responses_impl_with_conversations(
    mock_inference_api,
    mock_tool_groups_api,
    mock_tool_runtime_api,
    mock_responses_store,
    mock_vector_io_api,
    mock_conversations_api,
    mock_safety_api,
    mock_prompts_api,
    mock_files_api,
):
    """Create OpenAIResponsesImpl instance with conversations API."""
    return OpenAIResponsesImpl(
        inference_api=mock_inference_api,
        tool_groups_api=mock_tool_groups_api,
        tool_runtime_api=mock_tool_runtime_api,
        responses_store=mock_responses_store,
        vector_io_api=mock_vector_io_api,
        conversations_api=mock_conversations_api,
        safety_api=mock_safety_api,
        prompts_api=mock_prompts_api,
        files_api=mock_files_api,
    )


class TestConversationValidation:
    """Test conversation ID validation logic."""

    async def test_nonexistent_conversation_raises_error(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test that ConversationNotFoundError is raised for non-existent conversation."""
        conv_id = "conv_nonexistent"

        # Mock conversation not found
        mock_conversations_api.list_items.side_effect = ConversationNotFoundError("conv_nonexistent")

        with pytest.raises(ConversationNotFoundError):
            await responses_impl_with_conversations.create_openai_response(
                input="Hello", model="test-model", conversation=conv_id, stream=False
            )


class TestMessageSyncing:
    """Test message syncing to conversations."""

    async def test_sync_response_to_conversation_simple(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test syncing simple response to conversation."""
        conv_id = "conv_test123"
        input_text = "What are the 5 Ds of dodgeball?"

        # Output items (what the model generated)
        output_items = [
            OpenAIResponseMessage(
                id="msg_response",
                content=[
                    OpenAIResponseOutputMessageContentOutputText(
                        text="The 5 Ds are: Dodge, Duck, Dip, Dive, and Dodge.", type="output_text", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        await responses_impl_with_conversations._sync_response_to_conversation(conv_id, input_text, output_items)

        # should call add_items with user input and assistant response
        mock_conversations_api.add_items.assert_called_once()
        call_args = mock_conversations_api.add_items.call_args

        assert call_args[0][0] == conv_id  # conversation_id
        items = call_args[0][1]  # conversation_items

        assert len(items) == 2
        # User message
        assert items[0].type == "message"
        assert items[0].role == "user"
        assert items[0].content[0].type == "input_text"
        assert items[0].content[0].text == input_text

        # Assistant message
        assert items[1].type == "message"
        assert items[1].role == "assistant"

    async def test_sync_response_to_conversation_api_error(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        mock_conversations_api.add_items.side_effect = Exception("API Error")
        output_items = []

        # matching the behavior of OpenAI here
        with pytest.raises(Exception, match="API Error"):
            await responses_impl_with_conversations._sync_response_to_conversation(
                "conv_test123", "Hello", output_items
            )

    async def test_sync_with_list_input(self, responses_impl_with_conversations, mock_conversations_api):
        """Test syncing with list of input messages."""
        conv_id = "conv_test123"
        input_messages = [
            OpenAIResponseMessage(role="user", content=[{"type": "input_text", "text": "First message"}]),
        ]
        output_items = [
            OpenAIResponseMessage(
                id="msg_response",
                content=[OpenAIResponseOutputMessageContentOutputText(text="Response", type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        await responses_impl_with_conversations._sync_response_to_conversation(conv_id, input_messages, output_items)

        mock_conversations_api.add_items.assert_called_once()
        call_args = mock_conversations_api.add_items.call_args

        items = call_args[0][1]
        # Should have input message + output message
        assert len(items) == 2


class TestIntegrationWorkflow:
    """Integration tests for the full conversation workflow."""

    async def test_create_response_with_valid_conversation(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test creating a response with a valid conversation parameter."""
        mock_conversations_api.list_items.return_value = ConversationItemList(
            data=[], first_id=None, has_more=False, last_id=None, object="list"
        )

        async def mock_streaming_response(*args, **kwargs):
            message_item = OpenAIResponseMessage(
                id="msg_response",
                content=[
                    OpenAIResponseOutputMessageContentOutputText(
                        text="Test response", type="output_text", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )

            # Emit output_item.done event first (needed for conversation sync)
            yield OpenAIResponseObjectStreamResponseOutputItemDone(
                response_id="resp_test123",
                item=message_item,
                output_index=0,
                sequence_number=1,
                type="response.output_item.done",
            )

            # Then emit response.completed
            mock_response = OpenAIResponseObject(
                id="resp_test123",
                created_at=1234567890,
                model="test-model",
                object="response",
                output=[message_item],
                status="completed",
            )

            yield OpenAIResponseObjectStreamResponseCompleted(response=mock_response, type="response.completed")

        responses_impl_with_conversations._create_streaming_response = mock_streaming_response

        input_text = "Hello, how are you?"
        conversation_id = "conv_test123"

        response = await responses_impl_with_conversations.create_openai_response(
            input=input_text, model="test-model", conversation=conversation_id, stream=False
        )

        assert response is not None
        assert response.id == "resp_test123"

        # Note: conversation sync happens inside _create_streaming_response,
        # which we're mocking here, so we can't test it in this unit test.
        # The sync logic is tested separately in TestMessageSyncing.

    async def test_create_response_with_invalid_conversation_id(self, responses_impl_with_conversations):
        """Test creating a response with an invalid conversation ID."""
        with pytest.raises(InvalidConversationIdError) as exc_info:
            await responses_impl_with_conversations.create_openai_response(
                input="Hello", model="test-model", conversation="invalid_id", stream=False
            )

        assert "Expected an ID that begins with 'conv_'" in str(exc_info.value)

    async def test_create_response_with_nonexistent_conversation(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test creating a response with a non-existent conversation."""
        mock_conversations_api.list_items.side_effect = ConversationNotFoundError("conv_nonexistent")

        with pytest.raises(ConversationNotFoundError) as exc_info:
            await responses_impl_with_conversations.create_openai_response(
                input="Hello", model="test-model", conversation="conv_nonexistent", stream=False
            )

        assert "not found" in str(exc_info.value)

    async def test_conversation_and_previous_response_id(
        self, responses_impl_with_conversations, mock_conversations_api, mock_responses_store
    ):
        with pytest.raises(ValueError) as exc_info:
            await responses_impl_with_conversations.create_openai_response(
                input="test", model="test", conversation="conv_123", previous_response_id="resp_123"
            )

        assert "Mutually exclusive parameters" in str(exc_info.value)
        assert "previous_response_id" in str(exc_info.value)
        assert "conversation" in str(exc_info.value)
