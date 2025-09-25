# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for LlamaStackAsLibraryClient automatic initialization.

These tests ensure that the library client is automatically initialized
and ready to use immediately after construction.
"""

from llama_stack.core.library_client import (
    AsyncLlamaStackAsLibraryClient,
    LlamaStackAsLibraryClient,
)
from llama_stack.core.server.routes import RouteImpls


class TestLlamaStackAsLibraryClientAutoInitialization:
    """Test automatic initialization of library clients."""

    def test_sync_client_auto_initialization(self, monkeypatch):
        """Test that sync client is automatically initialized after construction."""
        # Mock the stack construction to avoid dependency issues
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("llama_stack.core.library_client.Stack", MockStack)
        monkeypatch.setattr("llama_stack.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = LlamaStackAsLibraryClient("ci-tests")

        assert client.async_client.route_impls is not None

    async def test_async_client_auto_initialization(self, monkeypatch):
        """Test that async client can be initialized and works properly."""
        # Mock the stack construction to avoid dependency issues
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("llama_stack.core.library_client.Stack", MockStack)
        monkeypatch.setattr("llama_stack.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncLlamaStackAsLibraryClient("ci-tests")

        # Initialize the client
        result = await client.initialize()
        assert result is True
        assert client.route_impls is not None

    def test_initialize_method_backward_compatibility(self, monkeypatch):
        """Test that initialize() method still works for backward compatibility."""
        # Mock the stack construction to avoid dependency issues
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("llama_stack.core.library_client.Stack", MockStack)
        monkeypatch.setattr("llama_stack.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = LlamaStackAsLibraryClient("ci-tests")

        result = client.initialize()
        assert result is None

        result2 = client.initialize()
        assert result2 is None

    async def test_async_initialize_method_idempotent(self, monkeypatch):
        """Test that async initialize() method can be called multiple times safely."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("llama_stack.core.library_client.Stack", MockStack)
        monkeypatch.setattr("llama_stack.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncLlamaStackAsLibraryClient("ci-tests")

        result1 = await client.initialize()
        assert result1 is True

        result2 = await client.initialize()
        assert result2 is True

    def test_route_impls_automatically_set(self, monkeypatch):
        """Test that route_impls is automatically set during construction."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("llama_stack.core.library_client.Stack", MockStack)
        monkeypatch.setattr("llama_stack.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        sync_client = LlamaStackAsLibraryClient("ci-tests")
        assert sync_client.async_client.route_impls is not None
