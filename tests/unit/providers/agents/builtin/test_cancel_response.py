# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for cancel_openai_response in Responses API."""

from unittest.mock import AsyncMock

import pytest

from llama_stack.providers.inline.agents.builtin.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.utils.responses.responses_store import (
    ResponsesStore,
    _OpenAIResponseObjectWithInputAndMessages,
)
from llama_stack_api import InvalidParameterError  # noqa: F401
from llama_stack_api.openai_responses import OpenAIResponseObject


@pytest.fixture
def mock_responses_store():
    return AsyncMock(spec=ResponsesStore)


@pytest.fixture
def openai_responses_impl(mock_responses_store):
    return OpenAIResponsesImpl(
        inference_api=AsyncMock(),
        tool_groups_api=AsyncMock(),
        tool_runtime_api=AsyncMock(),
        responses_store=mock_responses_store,
        vector_io_api=AsyncMock(),
        safety_api=AsyncMock(),
        conversations_api=AsyncMock(),
        prompts_api=AsyncMock(),
        files_api=AsyncMock(),
        connectors_api=AsyncMock(),
    )


def _make_stored_response(response_id: str, status: str, **kwargs) -> _OpenAIResponseObjectWithInputAndMessages:
    return _OpenAIResponseObjectWithInputAndMessages(
        id=response_id,
        created_at=1234567890,
        model="test-model",
        status=status,
        output=[],
        input=[],
        store=True,
        **kwargs,
    )


class TestCancelOpenAIResponse:
    """Tests for OpenAIResponsesImpl.cancel_openai_response."""

    async def test_cancel_queued_response(self, openai_responses_impl, mock_responses_store):
        """Cancelling a queued response sets status to cancelled."""
        stored = _make_stored_response("resp_1", "queued", background=True)
        mock_responses_store.get_response_object.return_value = stored

        result = await openai_responses_impl.cancel_openai_response("resp_1")

        assert isinstance(result, OpenAIResponseObject)
        assert result.status == "cancelled"
        assert result.id == "resp_1"
        mock_responses_store.update_response_object.assert_awaited_once()

    async def test_cancel_in_progress_response(self, openai_responses_impl, mock_responses_store):
        """Cancelling an in_progress response sets status to cancelled."""
        stored = _make_stored_response("resp_2", "in_progress")
        mock_responses_store.get_response_object.return_value = stored

        result = await openai_responses_impl.cancel_openai_response("resp_2")

        assert result.status == "cancelled"
        assert result.id == "resp_2"
        mock_responses_store.update_response_object.assert_awaited_once()

    async def test_cancel_already_cancelled_is_idempotent(self, openai_responses_impl, mock_responses_store):
        """Cancelling an already-cancelled response returns it without error."""
        stored = _make_stored_response("resp_3", "cancelled")
        mock_responses_store.get_response_object.return_value = stored

        result = await openai_responses_impl.cancel_openai_response("resp_3")

        assert result.status == "cancelled"
        assert result.id == "resp_3"
        # Should NOT call update since status is already cancelled
        mock_responses_store.update_response_object.assert_not_awaited()

    async def test_cancel_completed_response_raises(self, openai_responses_impl, mock_responses_store):
        """Cancelling a completed response raises InvalidParameterError."""
        stored = _make_stored_response("resp_4", "completed")
        mock_responses_store.get_response_object.return_value = stored

        with pytest.raises(ValueError, match="cannot be cancelled"):
            await openai_responses_impl.cancel_openai_response("resp_4")

        mock_responses_store.update_response_object.assert_not_awaited()

    async def test_cancel_failed_response_raises(self, openai_responses_impl, mock_responses_store):
        """Cancelling a failed response raises InvalidParameterError."""
        stored = _make_stored_response("resp_5", "failed")
        mock_responses_store.get_response_object.return_value = stored

        with pytest.raises(ValueError, match="cannot be cancelled"):
            await openai_responses_impl.cancel_openai_response("resp_5")

    async def test_cancel_incomplete_response_raises(self, openai_responses_impl, mock_responses_store):
        """Cancelling an incomplete response raises InvalidParameterError."""
        stored = _make_stored_response("resp_6", "incomplete")
        mock_responses_store.get_response_object.return_value = stored

        with pytest.raises(ValueError, match="cannot be cancelled"):
            await openai_responses_impl.cancel_openai_response("resp_6")

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "incomplete"])
    async def test_cancel_terminal_status_includes_current_status_in_error(
        self, openai_responses_impl, mock_responses_store, terminal_status
    ):
        """Error message includes the current status for clarity."""
        stored = _make_stored_response("resp_7", terminal_status)
        mock_responses_store.get_response_object.return_value = stored

        with pytest.raises(ValueError, match=terminal_status):
            await openai_responses_impl.cancel_openai_response("resp_7")
