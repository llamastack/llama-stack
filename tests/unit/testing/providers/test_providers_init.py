# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for provider registry and exception reconstruction dispatch."""

import httpx
from ollama import ResponseError
from openai import NotFoundError

from llama_stack.testing.exception_utils import GenericProviderError
from llama_stack.testing.providers import create_provider_error, detect_provider


class TestDetectProvider:
    """Test provider detection from exception module path."""

    def test_openai_exception_detected(self):
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        response = httpx.Response(404, request=request)
        exc = NotFoundError(message="x", response=response, body=None)
        assert detect_provider(exc) == "openai"

    def test_ollama_exception_detected(self):
        exc = ResponseError(error="model not found", status_code=404)
        assert detect_provider(exc) == "ollama"

    def test_unknown_exception_returns_unknown(self):
        exc = ValueError("plain Python exception")
        assert detect_provider(exc) == "unknown"


class TestCreateProviderError:
    """Test provider-specific error reconstruction."""

    def test_openai_reconstructs_specific_type_by_status(self):
        """404 -> NotFoundError, 429 -> RateLimitError, etc."""
        exc = create_provider_error("openai", 404, {"error": {"code": "not_found"}}, "Not found")
        assert isinstance(exc, NotFoundError)
        assert exc.status_code == 404
        assert "not found" in str(exc).lower()

    def test_openai_unknown_status_falls_back_to_api_status_error(self):
        """Unmapped status codes still produce valid APIStatusError."""
        exc = create_provider_error("openai", 418, None, "I'm a teapot")
        assert exc.status_code == 418
        assert "teapot" in str(exc).lower()

    def test_ollama_reconstructs_response_error(self):
        exc = create_provider_error("ollama", 404, None, "model not found")
        assert isinstance(exc, ResponseError)
        assert exc.status_code == 404
        assert "not found" in str(exc).lower()

    def test_unknown_provider_returns_generic_with_status_and_body(self):
        """Unknown providers get GenericProviderError for consistent replay."""
        exc = create_provider_error("future_sdk", 503, {"retry_after": 60}, "Unavailable")
        assert isinstance(exc, GenericProviderError)
        assert exc.status_code == 503
        assert exc.body == {"retry_after": 60}
        assert "unavailable" in str(exc).lower()
