# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for streaming error extraction functions."""

from unittest.mock import MagicMock

from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
    extract_openai_error,
)


def _make_mock_exc(body):
    """Create a mock APIStatusError with the given body."""
    exc = MagicMock()
    exc.body = body
    exc.configure_mock(**{"__str__.return_value": f"Error code: 400 - {body}"})
    return exc


class TestExtractOpenaiError:
    """Tests for extract_openai_error function.

    The OpenAI SDK provides errors in two formats:
        1. Nested: {"error": {"code": "...", "message": "...", ...}}
        2. Direct: {"code": "...", "message": "...", ...}

    When "code" is missing or empty, falls back to "server_error".
    The message is always preserved so users get the real error details.
    """

    def test_nested_format_extracts_correctly(self):
        """Nested format: {"error": {"code": "...", "message": "..."}}."""
        body = {"error": {"code": "invalid_image_url", "message": "Failed to download image"}}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "invalid_image_url"
        assert message == "Failed to download image"

    def test_direct_format_extracts_correctly(self):
        """Direct format: {"code": "...", "message": "..."}."""
        body = {"code": "invalid_image_url", "message": "Failed to download image", "type": "invalid_request_error"}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "invalid_image_url"
        assert message == "Failed to download image"

    def test_maps_invalid_base64_to_responses_api_code(self):
        """Chat Completions 'invalid_base64' maps to Responses API 'invalid_base64_image'."""
        body = {"error": {"code": "invalid_base64", "message": "Invalid base64 data"}}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "invalid_base64_image"
        assert message == "Invalid base64 data"

    def test_direct_format_maps_codes(self):
        """Direct format also maps codes correctly."""
        body = {"code": "invalid_base64", "message": "Invalid base64 data"}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "invalid_base64_image"
        assert message == "Invalid base64 data"

    def test_missing_message_uses_str_exc(self):
        """Missing message field uses str(exc) as fallback."""
        body = {"code": "invalid_image_url"}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "invalid_image_url"
        assert "Error code: 400" in message

    def test_nested_error_not_dict_falls_back_to_direct(self):
        """If 'error' key exists but isn't a dict, try direct format."""
        body = {"error": "string error", "code": "rate_limit_exceeded", "message": "Too many requests"}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "rate_limit_exceeded"
        assert message == "Too many requests"

    def test_missing_code_returns_server_error_preserves_message(self):
        """Missing code returns server_error but preserves the message."""
        body = {"message": "Model does not support images", "type": "invalid_request_error", "code": None}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "server_error"
        assert message == "Model does not support images"

    def test_none_body_returns_server_error(self):
        """None body returns server_error."""
        exc = _make_mock_exc(body=None)
        code, message = extract_openai_error(exc)
        assert code == "server_error"
        assert "Error code: 400" in message

    def test_non_dict_body_returns_server_error(self):
        """Non-dict body (e.g., string) returns server_error."""
        exc = _make_mock_exc(body="unexpected string body")
        code, message = extract_openai_error(exc)
        assert code == "server_error"

    def test_no_code_preserves_message(self):
        """Missing code returns server_error but preserves message."""
        body = {"message": "Something went wrong"}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "server_error"
        assert message == "Something went wrong"

    def test_empty_string_code_returns_server_error(self):
        """Empty string code returns server_error but preserves message."""
        body = {"code": "", "message": "Bad request", "type": "invalid_request_error"}
        exc = _make_mock_exc(body=body)
        code, message = extract_openai_error(exc)
        assert code == "server_error"
        assert message == "Bad request"
