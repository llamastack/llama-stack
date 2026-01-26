# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for background mode in the Responses API."""

import time

import pytest


class TestBackgroundResponses:
    """Test background mode for response generation."""

    def test_background_response_returns_queued(self, responses_client, text_model_id):
        """Test that background=True returns immediately with queued status."""
        response = responses_client.responses.create(
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

    def test_background_response_completes(self, responses_client, text_model_id):
        """Test that a background response eventually completes."""
        response = responses_client.responses.create(
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

            retrieved = responses_client.responses.retrieve(response_id=response_id)

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

    def test_cancel_background_response(self, responses_client, text_model_id):
        """Test cancelling a background response."""
        # Create a background response with a longer prompt to give time to cancel
        response = responses_client.responses.create(
            model=text_model_id,
            input="Write a very long and detailed story about a dragon who learns to code. Include many chapters.",
            background=True,
        )

        assert response.status == "queued"
        response_id = response.id

        # Cancel the response
        cancelled = responses_client.responses.cancel(response_id=response_id)

        assert cancelled.status == "cancelled"
        assert cancelled.background is True
        assert cancelled.id == response_id

    def test_cannot_cancel_non_background_response(self, responses_client, text_model_id):
        """Test that non-background responses cannot be cancelled."""
        response = responses_client.responses.create(
            model=text_model_id,
            input="Hello",
            background=False,
        )

        assert response.status == "completed"
        assert response.background is False

        # Attempting to cancel should raise an error
        with pytest.raises(Exception) as exc_info:
            responses_client.responses.cancel(response_id=response.id)

        assert "cannot be cancelled" in str(exc_info.value).lower()

    def test_cannot_cancel_completed_background_response(self, responses_client, text_model_id):
        """Test that completed background responses cannot be cancelled."""
        response = responses_client.responses.create(
            model=text_model_id,
            input="Say hi",
            background=True,
        )

        response_id = response.id

        # Wait for completion
        max_wait = 60
        poll_interval = 1
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            retrieved = responses_client.responses.retrieve(response_id=response_id)
            if retrieved.status == "completed":
                break

        # Now try to cancel the completed response
        with pytest.raises(Exception) as exc_info:
            responses_client.responses.cancel(response_id=response_id)

        assert "cannot be cancelled" in str(exc_info.value).lower()

    def test_background_and_stream_mutually_exclusive(self, responses_client, text_model_id):
        """Test that background=True and stream=True cannot be used together."""
        with pytest.raises(Exception) as exc_info:
            responses_client.responses.create(
                model=text_model_id,
                input="Hello",
                background=True,
                stream=True,
            )

        error_msg = str(exc_info.value).lower()
        assert "background" in error_msg or "stream" in error_msg

    def test_background_false_is_synchronous(self, responses_client, text_model_id):
        """Test that background=False returns a completed response synchronously."""
        response = responses_client.responses.create(
            model=text_model_id,
            input="What is 1+1?",
            background=False,
        )

        assert response.status == "completed"
        assert response.background is False
        assert len(response.output) > 0
