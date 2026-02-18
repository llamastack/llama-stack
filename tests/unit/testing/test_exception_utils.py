# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for exception serialization/deserialization used in API recording replay."""

import httpx
from ollama import ResponseError
from openai import NotFoundError

from llama_stack.core.exceptions.translation import translate_exception
from llama_stack.testing.exception_utils import (
    deserialize_exception,
    is_provider_sdk_exception,
    serialize_exception,
)
from llama_stack_api.common.errors import (
    BatchNotFoundError,
    ConflictError,
    LlamaStackError,
    ModelNotFoundError,
)


class TestSerializeException:
    """Test exception categorization and serialization for recording."""

    def test_llama_stack_error_serializes_as_llama_stack_category(self):
        """LlamaStackError subclasses are categorized for accurate replay."""
        exc = ModelNotFoundError("llama-3")
        data = serialize_exception(exc)
        assert data["category"] == "llama_stack"
        assert data["type"] == "ModelNotFoundError"
        assert "llama-3" in data["message"]
        assert data["status_code"] == 404

    def test_provider_sdk_exception_serializes_with_provider_and_body(self):
        """OpenAI/Ollama exceptions are categorized with provider for reconstruction."""
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(404, json={"error": {"code": "not_found"}}, request=request)
        exc = NotFoundError(
            message="Batch not found",
            response=response,
            body={"error": {"code": "not_found"}},
        )
        data = serialize_exception(exc)
        assert data["category"] == "provider_sdk"
        assert data["provider"] == "openai"
        assert data["status_code"] == 404
        assert data["body"] == {"error": {"code": "not_found"}}
        assert "not found" in data["message"].lower()

    def test_builtin_exception_serializes_by_type_name(self):
        """Mapped built-in exceptions allow reconstruction from type name."""
        exc = ValueError("invalid input")
        data = serialize_exception(exc)
        assert data["category"] == "builtin"
        assert data["type"] == "ValueError"
        assert data["message"] == "invalid input"

    def test_unknown_exception_serializes_with_type_and_message(self):
        """Unmapped exceptions still capture type and message for generic replay."""
        exc = RuntimeError("unexpected failure")
        data = serialize_exception(exc)
        assert data["category"] == "unknown"
        assert data["type"] == "RuntimeError"
        assert data["message"] == "unexpected failure"


class TestDeserializeException:
    """Test exception reconstruction from recorded data."""

    def test_llama_stack_roundtrip_preserves_status_and_message(self):
        """Reconstructed LlamaStackError has interface needed for translate_exception."""
        exc = BatchNotFoundError("batch-xyz")
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert isinstance(reconstructed, LlamaStackError)
        assert reconstructed.status_code == 404
        assert "batch-xyz" in str(reconstructed)

    def test_provider_sdk_roundtrip_preserves_type_and_status(self):
        """OpenAI exceptions reconstruct to same type for client compatibility."""
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(404, json={"error": {"code": "invalid_request_error"}}, request=request)
        exc = NotFoundError(
            message="Resource not found",
            response=response,
            body={"error": {"code": "invalid_request_error"}},
        )
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert isinstance(reconstructed, NotFoundError)
        assert reconstructed.status_code == 404
        assert "not found" in str(reconstructed).lower()

    def test_builtin_roundtrip_reconstructs_exact_type(self):
        """Built-in exceptions reconstruct to original type."""
        exc = ValueError("bad value")
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert type(reconstructed) is ValueError
        assert str(reconstructed) == "bad value"

    def test_unknown_roundtrip_falls_back_to_generic_exception(self):
        """Unknown exceptions replay as generic Exception with message preserved."""
        exc = RuntimeError("internal error")
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert type(reconstructed) is Exception
        assert str(reconstructed) == "internal error"

    def test_deserialize_missing_category_defaults_to_unknown(self):
        """Legacy or malformed data without category still produces usable exception."""
        data = {"type": "RuntimeError", "message": "legacy format"}
        reconstructed = deserialize_exception(data)
        assert type(reconstructed) is Exception
        assert str(reconstructed) == "legacy format"


class TestReconstructedExceptionInterface:
    """Verify reconstructed exceptions work with server's translate_exception."""

    def test_llama_stack_reconstructed_translates_to_http(self):
        """Reconstructed LlamaStackError produces correct HTTP status via translate_exception."""
        data = {"category": "llama_stack", "status_code": 404, "message": "Batch xyz not found"}
        exc = deserialize_exception(data)
        http_exc = translate_exception(exc)
        assert http_exc.status_code == 404
        assert "xyz" in http_exc.detail

    def test_provider_sdk_reconstructed_translates_to_http(self):
        """Reconstructed provider errors produce correct HTTP status."""
        data = {
            "category": "provider_sdk",
            "provider": "openai",
            "status_code": 429,
            "message": "Rate limit exceeded",
            "body": None,
        }
        exc = deserialize_exception(data)
        http_exc = translate_exception(exc)
        assert http_exc.status_code == 429

    def test_unknown_provider_uses_generic_but_preserves_status(self):
        """Unknown provider falls back to GenericProviderError with status_code."""
        data = {
            "category": "provider_sdk",
            "provider": "future_sdk",
            "status_code": 503,
            "message": "Service unavailable",
            "body": None,
        }
        exc = deserialize_exception(data)
        assert hasattr(exc, "status_code")
        assert exc.status_code == 503
        http_exc = translate_exception(exc)
        assert http_exc.status_code == 503


class TestIsProviderSdkException:
    """Test provider SDK exception detection used during serialization."""

    def test_openai_exception_detected(self):
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        response = httpx.Response(404, request=request)
        assert is_provider_sdk_exception(NotFoundError(message="x", response=response, body=None))

    def test_ollama_exception_detected(self):
        assert is_provider_sdk_exception(ResponseError(error="x", status_code=404))

    def test_llama_stack_not_detected_as_provider_sdk(self):
        """LlamaStackError has status_code but is handled separately (llama_stack category)."""
        exc = ConflictError("conflict")
        # serialize_exception checks LlamaStackError first, so we never get to provider_sdk
        data = serialize_exception(exc)
        assert data["category"] == "llama_stack"

    def test_plain_exception_not_detected(self):
        assert not is_provider_sdk_exception(ValueError("x"))
