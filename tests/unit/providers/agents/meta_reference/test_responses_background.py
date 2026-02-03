# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for background parameter support in Responses API."""

import pytest

from llama_stack_api import OpenAIResponseError, OpenAIResponseObject


class TestBackgroundFieldInResponseObject:
    """Test that the background field is properly defined in OpenAIResponseObject."""

    def test_background_field_default_is_none(self):
        """Verify background field defaults to None."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="completed",
            output=[],
            store=True,
        )
        assert response.background is None

    def test_background_field_can_be_true(self):
        """Verify background field can be set to True."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="queued",
            output=[],
            background=True,
            store=True,
        )
        assert response.background is True

    def test_background_field_can_be_false(self):
        """Verify background field can be False."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="completed",
            output=[],
            background=False,
            store=True,
        )
        assert response.background is False


class TestResponseStatus:
    """Test that all expected status values work correctly."""

    @pytest.mark.parametrize(
        "status",
        ["queued", "in_progress", "completed", "failed", "incomplete"],
    )
    def test_valid_status_values(self, status):
        """Verify all OpenAI-compatible status values are accepted."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status=status,
            output=[],
            background=True if status in ("queued", "in_progress") else False,
            store=True,
        )
        assert response.status == status

    def test_queued_status_with_background(self):
        """Verify queued status is typically used with background=True."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="queued",
            output=[],
            background=True,
            store=True,
        )
        assert response.status == "queued"
        assert response.background is True


class TestResponseObjectSerialization:
    """Test that the response object serializes correctly with background field."""

    def test_model_dump_includes_background(self):
        """Verify model_dump includes the background field."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="queued",
            output=[],
            background=True,
            store=True,
        )
        data = response.model_dump()
        assert "background" in data
        assert data["background"] is True

    def test_model_dump_json_includes_background(self):
        """Verify JSON serialization includes the background field."""
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="completed",
            output=[],
            background=False,
            store=True,
        )
        json_str = response.model_dump_json()
        assert '"background":false' in json_str or '"background": false' in json_str


class TestResponseErrorForBackground:
    """Test error responses for background processing failures."""

    def test_error_response_with_background(self):
        """Verify error responses can include background field."""
        error = OpenAIResponseError(
            code="processing_error",
            message="Background processing failed",
        )
        response = OpenAIResponseObject(
            id="resp_123",
            created_at=1234567890,
            model="test-model",
            status="failed",
            output=[],
            background=True,
            error=error,
            store=True,
        )
        assert response.status == "failed"
        assert response.background is True
        assert response.error is not None
        assert response.error.code == "processing_error"
