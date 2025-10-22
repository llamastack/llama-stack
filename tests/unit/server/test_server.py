# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

from fastapi import HTTPException
from openai import BadRequestError
from pydantic import ValidationError

from llama_stack.core.access_control.access_control import AccessDeniedError
from llama_stack.core.datatypes import AuthenticationRequiredError
from llama_stack.core.server.server import translate_exception


class TestTranslateException:
    """Test cases for the translate_exception function."""

    def test_translate_access_denied_error(self):
        """Test that AccessDeniedError is translated to 403 HTTP status."""
        exc = AccessDeniedError()
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 403
        assert result.detail == "Permission denied: Insufficient permissions"

    def test_translate_access_denied_error_with_context(self):
        """Test that AccessDeniedError with context includes detailed information."""
        from llama_stack.core.datatypes import User

        # Create mock user and resource
        user = User("test-user", {"roles": ["user"], "teams": ["dev"]})

        # Create a simple mock object that implements the ProtectedResource protocol
        class MockResource:
            def __init__(self, type: str, identifier: str, owner=None):
                self.type = type
                self.identifier = identifier
                self.owner = owner

        resource = MockResource("vector_store", "test-db")

        exc = AccessDeniedError("create", resource, user)
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 403
        assert "test-user" in result.detail
        assert "vector_store::test-db" in result.detail
        assert "create" in result.detail
        assert "roles=['user']" in result.detail
        assert "teams=['dev']" in result.detail

    def test_translate_permission_error(self):
        """Test that PermissionError is translated to 403 HTTP status."""
        exc = PermissionError("Permission denied")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 403
        assert result.detail == "Permission denied: Permission denied"

    def test_translate_value_error(self):
        """Test that ValueError is translated to 400 HTTP status."""
        exc = ValueError("Invalid input")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 400
        assert result.detail == "Invalid value: Invalid input"

    def test_translate_bad_request_error(self):
        """Test that BadRequestError is translated to 400 HTTP status."""
        # Create a mock response for BadRequestError
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        exc = BadRequestError("Bad request", response=mock_response, body="Bad request")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 400
        assert result.detail == "Bad request"

    def test_translate_authentication_required_error(self):
        """Test that AuthenticationRequiredError is translated to 401 HTTP status."""
        exc = AuthenticationRequiredError("Authentication required")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 401
        assert result.detail == "Authentication required: Authentication required"

    def test_translate_timeout_error(self):
        """Test that TimeoutError is translated to 504 HTTP status."""
        exc = TimeoutError("Operation timed out")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 504
        assert result.detail == "Operation timed out: Operation timed out"

    def test_translate_asyncio_timeout_error(self):
        """Test that asyncio.TimeoutError is translated to 504 HTTP status."""
        exc = TimeoutError()
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 504
        assert result.detail == "Operation timed out: "

    def test_translate_connection_error(self):
        """Test that ConnectionError is translated to 502 HTTP status."""
        exc = ConnectionError("Failed to connect to MCP server at http://localhost:9999/sse: Connection refused")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 502
        assert result.detail == "Failed to connect to MCP server at http://localhost:9999/sse: Connection refused"

    def test_translate_not_implemented_error(self):
        """Test that NotImplementedError is translated to 501 HTTP status."""
        exc = NotImplementedError("Not implemented")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 501
        assert result.detail == "Not implemented: Not implemented"

    def test_translate_validation_error(self):
        """Test that ValidationError is translated to 400 HTTP status with proper format."""
        # Create a mock validation error using proper Pydantic error format
        exc = ValidationError.from_exception_data(
            "TestModel",
            [
                {
                    "loc": ("field", "nested"),
                    "msg": "field required",
                    "type": "missing",
                }
            ],
        )

        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 400
        assert "errors" in result.detail
        assert len(result.detail["errors"]) == 1
        assert result.detail["errors"][0]["loc"] == ["field", "nested"]
        assert result.detail["errors"][0]["msg"] == "Field required"
        assert result.detail["errors"][0]["type"] == "missing"

    def test_translate_generic_exception(self):
        """Test that generic exceptions are translated to 500 HTTP status."""
        exc = Exception("Unexpected error")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert result.detail == "Internal server error: An unexpected error occurred."

    def test_translate_runtime_error(self):
        """Test that RuntimeError is translated to 500 HTTP status."""
        exc = RuntimeError("Runtime error")
        result = translate_exception(exc)

        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert result.detail == "Internal server error: An unexpected error occurred."

    def test_multiple_access_denied_scenarios(self):
        """Test various scenarios that should result in 403 status codes."""
        # Test AccessDeniedError (uses enhanced message)
        exc1 = AccessDeniedError()
        result1 = translate_exception(exc1)
        assert isinstance(result1, HTTPException)
        assert result1.status_code == 403
        assert result1.detail == "Permission denied: Insufficient permissions"

        # Test PermissionError (uses generic message)
        exc2 = PermissionError("No permission")
        result2 = translate_exception(exc2)
        assert isinstance(result2, HTTPException)
        assert result2.status_code == 403
        assert result2.detail == "Permission denied: No permission"

        exc3 = PermissionError("Access denied")
        result3 = translate_exception(exc3)
        assert isinstance(result3, HTTPException)
        assert result3.status_code == 403
        assert result3.detail == "Permission denied: Access denied"
