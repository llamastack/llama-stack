# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest


@pytest.mark.integration
class TestOpenAIResponses:
    """Integration tests for the OpenAI responses API."""

    def test_openai_response_with_max_output_tokens(self, openai_client, text_model_id):
        """Test OpenAI response with max_output_tokens parameter."""
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What are the 5 Ds of dodgeball?"}],
            max_output_tokens=100,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.max_output_tokens == 100

    def test_openai_response_with_small_max_output_tokens(self, openai_client, text_model_id):
        """Test response with very small max_output_tokens to trigger potential truncation."""
        response = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "role": "user",
                    "content": "Write a detailed essay about the history of artificial intelligence, covering the past 70 years.",
                }
            ],
            max_output_tokens=20,
        )

        assert response.id.startswith("resp_")
        assert response.max_output_tokens == 20
        assert len(response.output_text.strip()) > 0

        # With such a small token limit, the response might be incomplete
        # Note: The status might be 'incomplete' depending on provider implementation
        if response.usage is not None and response.usage.output_tokens > 0:
            # Allow some tolerance for provider differences
            assert response.usage.output_tokens <= 25, (
                f"Output tokens ({response.usage.output_tokens}) should respect max_output_tokens (20) "
            )

    def test_openai_response_max_output_tokens_below_minimum(self, openai_client, text_model_id):
        """Test that max_output_tokens below minimum (< 16) is rejected."""
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                max_output_tokens=15,
            )

        # Should get a validation error
        error_message = str(exc_info.value).lower()
        assert "validation" in error_message or "invalid" in error_message or "16" in error_message
