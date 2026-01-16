# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Error handling tests for the Llama Stack Responses and Conversations APIs.

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
from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI


class TestResponsesAPIErrors:
    """Error handling tests for the Responses API.

    These tests verify SDK compatibility by ensuring Llama Stack returns
    the correct HTTP status codes that trigger the expected OpenAI SDK
    exception types for Responses API operations.
    """

    def test_invalid_model_raises_not_found_error(self, openai_client):
        """
        Test that requesting a nonexistent model returns 404 and triggers
        openai.NotFoundError in the SDK.

        This is critical for SDK compatibility - users catching NotFoundError
        should have their error handling work correctly.
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.create(
                model="nonexistent-model-xyz-12345",
                input="Hello, world!",
            )

        assert exc_info.value.status_code == 404
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "model" in error_msg

    def test_invalid_previous_response_id_raises_not_found_error(self, openai_client, text_model_id):
        """
        Test that referencing a nonexistent previous_response_id returns 404.

        Per OpenResponses spec, previous_response_id references a prior response
        for multi-turn conversations.
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Continue the conversation",
                previous_response_id="resp_nonexistent123456",
            )

        assert exc_info.value.status_code == 404
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "response" in error_msg

    def test_invalid_max_tool_calls_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that invalid max_tool_calls (< 1) returns 400.
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Search for news",
                tools=[{"type": "web_search"}],
                max_tool_calls=0,  # Invalid: must be >= 1
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "max_tool_calls" in error_msg or "invalid" in error_msg

    def test_invalid_temperature_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that temperature outside valid range (0-2) returns 400.

        Per OpenResponses spec: "Sampling temperature to use, between 0 and 2."
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                temperature=3.0,  # Invalid: must be between 0 and 2
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "temperature" in error_msg or "invalid" in error_msg or "range" in error_msg

    def test_invalid_top_p_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that top_p outside valid range (0-1) returns 400.

        Per OpenResponses spec: "Nucleus sampling parameter, between 0 and 1."
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                top_p=1.5,  # Invalid: must be between 0 and 1
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "top_p" in error_msg or "invalid" in error_msg or "range" in error_msg

    def test_invalid_tool_choice_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that invalid tool_choice value returns 400.

        Per OpenResponses spec, tool_choice controls which tool the model should use.
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
                tool_choice="invalid_choice",  # Invalid: must be valid enum or object
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "tool_choice" in error_msg or "invalid" in error_msg


class TestConversationsAPIErrors:
    """Error handling tests for the Conversations API.

    These tests verify SDK compatibility for conversation-related operations
    accessed through the Responses API.
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


class TestAuthenticationErrors:
    """Authentication error handling tests.

    These tests verify that authentication failures return the correct
    HTTP status codes that trigger openai.AuthenticationError in the SDK.
    """

    def test_invalid_api_key_raises_authentication_error(self, openai_client, text_model_id):
        """
        Test that an invalid API key returns 401 and triggers
        openai.AuthenticationError in the SDK.
        """
        # Create a client with an invalid API key
        unauthenticated_client = OpenAI(
            base_url=openai_client.base_url,
            api_key="invalid-api-key-xyz",
            max_retries=0,
            timeout=30.0,
        )

        try:
            with pytest.raises(AuthenticationError) as exc_info:
                unauthenticated_client.responses.create(
                    model=text_model_id,
                    input="Hello, world!",
                )

            assert exc_info.value.status_code == 401
        finally:
            unauthenticated_client.close()
