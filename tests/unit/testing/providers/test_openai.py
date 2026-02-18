# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for OpenAI provider exception reconstruction."""

from openai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    RateLimitError,
)

from llama_stack.testing.providers.openai import create_error


class TestOpenAICreateError:
    """Test OpenAI-specific error reconstruction for replay."""

    def test_status_code_maps_to_correct_class(self):
        """Each mapped status code reconstructs to the expected OpenAI error type."""
        cases = [
            (400, BadRequestError),
            (401, AuthenticationError),
            (404, NotFoundError),
            (409, ConflictError),
            (429, RateLimitError),
        ]
        for status, expected_class in cases:
            exc = create_error(status, None, f"error {status}")
            assert isinstance(exc, expected_class)
            assert exc.status_code == status

    def test_body_preserved_for_client_consumption(self):
        """Error body is attached for tests that inspect structured error data."""
        body = {"error": {"code": "invalid_request_error", "param": "model"}}
        exc = create_error(400, body, "Invalid model")
        assert exc.body == body

    def test_unmapped_status_uses_api_status_error_base(self):
        """Status codes without specific mapping still produce valid API errors."""
        exc = create_error(503, None, "Service unavailable")
        assert isinstance(exc, APIStatusError)
        assert exc.status_code == 503
