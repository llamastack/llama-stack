# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, Mock

from fastapi import Request

from llama_stack.core.server.query_params_middleware import QueryParamsMiddleware


class TestQueryParamsMiddleware:
    """Test cases for the QueryParamsMiddleware."""

    async def test_extracts_query_params_for_vector_store_content(self):
        """Test that middleware extracts query params for vector store content endpoints."""
        middleware = QueryParamsMiddleware(Mock())
        request = Mock(spec=Request)
        request.method = "GET"

        # Mock the URL properly
        mock_url = Mock()
        mock_url.path = "/v1/vector_stores/vs_123/files/file_456/content"
        request.url = mock_url

        request.query_params = {"include_embeddings": "true", "include_metadata": "false"}

        # Create a fresh state object without any attributes
        class MockState:
            pass

        request.state = MockState()

        await middleware.dispatch(request, AsyncMock())

        assert hasattr(request.state, "extra_query")
        assert request.state.extra_query == {"include_embeddings": True, "include_metadata": False}

    async def test_ignores_non_vector_store_endpoints(self):
        """Test that middleware ignores non-vector store endpoints."""
        middleware = QueryParamsMiddleware(Mock())
        request = Mock(spec=Request)
        request.method = "GET"

        # Mock the URL properly
        mock_url = Mock()
        mock_url.path = "/v1/inference/chat_completion"
        request.url = mock_url

        request.query_params = {"include_embeddings": "true"}

        # Create a fresh state object without any attributes
        class MockState:
            pass

        request.state = MockState()

        await middleware.dispatch(request, AsyncMock())

        assert not hasattr(request.state, "extra_query")

    async def test_handles_json_parsing(self):
        """Test that middleware correctly parses JSON values and handles invalid JSON."""
        middleware = QueryParamsMiddleware(Mock())
        request = Mock(spec=Request)
        request.method = "GET"

        # Mock the URL properly
        mock_url = Mock()
        mock_url.path = "/v1/vector_stores/vs_123/files/file_456/content"
        request.url = mock_url

        request.query_params = {"config": '{"key": "value"}', "invalid": "not-json{", "number": "42"}

        # Create a fresh state object without any attributes
        class MockState:
            pass

        request.state = MockState()

        await middleware.dispatch(request, AsyncMock())

        expected = {"config": {"key": "value"}, "invalid": "not-json{", "number": 42}
        assert request.state.extra_query == expected
