# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for background mode in the Responses API."""

import time

import pytest


@pytest.mark.integration
class TestBackgroundResponses:
    """Test background mode for response generation."""

    def test_background_response_returns_queued(self, openai_client, text_model_id):
        """Test that background=True returns immediately with queued status."""
        response = openai_client.responses.create(
            model=text_model_id,
            input="What is 2+2?",
            background=True,
        )

        # Should return immediately with queued status
        assert response.status == "queued"
        assert response.background is True
        assert response.id.startswith("resp_")
        # Output should be empty initially
        assert len(response.output) == 0

    def test_background_response_completes(self, openai_client, text_model_id):
        """Test that a background response eventually completes."""
        response = openai_client.responses.create(
            model=text_model_id,
            input="Say hello",
            background=True,
        )

        assert response.status == "queued"
        response_id = response.id

        # Poll for completion (max 60 seconds)
        max_wait = 60
        poll_interval = 1
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            retrieved = openai_client.responses.retrieve(response_id=response_id)

            if retrieved.status == "completed":
                assert retrieved.background is True
                assert len(retrieved.output) > 0
                assert len(retrieved.output_text) > 0
                return

            if retrieved.status == "failed":
                pytest.fail(f"Background response failed: {retrieved.error}")

            # Status should be queued or in_progress while processing
            assert retrieved.status in ("queued", "in_progress")

        pytest.fail(f"Background response did not complete within {max_wait} seconds")

    def test_background_and_stream_mutually_exclusive(self, openai_client, text_model_id):
        """Test that background=True and stream=True cannot be used together."""
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                background=True,
                stream=True,
            )

        error_msg = str(exc_info.value).lower()
        assert "background" in error_msg or "stream" in error_msg

    def test_background_false_is_synchronous(self, openai_client, text_model_id):
        """Test that background=False returns a completed response synchronously."""
        response = openai_client.responses.create(
            model=text_model_id,
            input="What is 1+1?",
            background=False,
        )

        assert response.status == "completed"
        assert response.background is False
        assert len(response.output) > 0

    def test_cancel_queued_or_in_progress_response(self, openai_client, text_model_id):
        """Test cancelling a background response that is queued or in progress."""
        # Create a background response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Write a detailed 5000 word essay about quantum physics and the nature of reality.",
            background=True,
        )

        assert response.status == "queued"
        response_id = response.id

        # Give it a moment to potentially start processing
        time.sleep(0.5)

        # Cancel the response
        cancelled = openai_client.responses.cancel(response_id=response_id)

        assert cancelled.id == response_id
        assert cancelled.status == "cancelled"
        # Note: background field may not be preserved through OpenAI client deserialization

        # Verify the response stays cancelled
        retrieved = openai_client.responses.retrieve(response_id=response_id)
        assert retrieved.status == "cancelled"

    def test_cancel_already_cancelled_is_idempotent(self, openai_client, text_model_id):
        """Test that cancelling an already-cancelled response is idempotent."""
        # Create and cancel a background response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Write a long story.",
            background=True,
        )

        response_id = response.id
        cancelled = openai_client.responses.cancel(response_id=response_id)
        assert cancelled.status == "cancelled"

        # Cancel again - should return same state without error
        cancelled_again = openai_client.responses.cancel(response_id=response_id)
        assert cancelled_again.id == response_id
        assert cancelled_again.status == "cancelled"

    def test_cancel_completed_response_fails(self, openai_client, text_model_id):
        """Test that cancelling a completed response returns 409 Conflict."""
        # Create a synchronous (completed) response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Say hello",
            background=False,
        )

        assert response.status == "completed"
        response_id = response.id

        # Try to cancel it - should fail with 409
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.cancel(response_id=response_id)

        # Check for conflict error (different clients may raise different exceptions)
        error_str = str(exc_info.value).lower()
        assert "409" in error_str or "conflict" in error_str or "cannot cancel" in error_str

    def test_cancel_nonexistent_response_fails(self, openai_client, text_model_id):
        """Test that cancelling a non-existent response returns 404."""
        fake_id = "resp_fake_nonexistent_id"

        with pytest.raises(Exception) as exc_info:
            openai_client.responses.cancel(response_id=fake_id)

        # Check for not found error
        error_str = str(exc_info.value).lower()
        assert "404" in error_str or "not found" in error_str

    def test_cancel_prevents_completion(self, openai_client, text_model_id):
        """Test that a cancelled response does not complete."""
        # Create a background response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Write a detailed essay.",
            background=True,
        )

        response_id = response.id
        assert response.status == "queued"

        # Cancel immediately
        cancelled = openai_client.responses.cancel(response_id=response_id)
        assert cancelled.status == "cancelled"

        # Wait a bit to ensure it doesn't complete
        time.sleep(3)

        # Verify it's still cancelled (not completed)
        retrieved = openai_client.responses.retrieve(response_id=response_id)
        assert retrieved.status == "cancelled"
        # Output should be empty since processing was cancelled
        assert len(retrieved.output) == 0
