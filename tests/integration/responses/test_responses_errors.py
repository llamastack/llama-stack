# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Error handling tests for the Llama Stack Responses API.

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


class TestResponsesAPIErrors:
    """Error handling tests for the Responses API using OpenAI client.

    These tests verify SDK compatibility by ensuring Llama Stack returns
    the correct HTTP status codes that trigger the expected OpenAI SDK
    exception types.
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

    def test_invalid_max_tool_calls_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that invalid parameter values return 400 and trigger
        openai.BadRequestError in the SDK.
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
