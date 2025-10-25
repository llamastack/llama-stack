# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import LlamaStackClient

from llama_stack import LlamaStackAsLibraryClient

pytestmark = pytest.mark.skip(reason="Requires client SDK update for new provider management APIs")


class TestDynamicProviderManagement:
    """Integration tests for dynamic provider registration, update, and unregistration."""

    def test_register_and_unregister_inference_provider(
        self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient
    ):
        """Test registering and unregistering an inference provider."""
        provider_id = "test-dynamic-inference"

        # Clean up if exists from previous test
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register a new inference provider (using Ollama since it's available in test setup)
        response = llama_stack_client.providers.register(
            provider_id=provider_id,
            api="inference",
            provider_type="remote::ollama",
            config={
                "url": "http://localhost:11434",
                "api_token": "",
            },
        )

        # Verify registration
        assert response.provider.provider_id == provider_id
        assert response.provider.api == "inference"
        assert response.provider.provider_type == "remote::ollama"
        assert response.provider.status in ["connected", "initializing"]

        # Verify provider appears in list
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id in provider_ids

        # Verify we can retrieve it
        provider = llama_stack_client.providers.retrieve(provider_id)
        assert provider.provider_id == provider_id

        # Unregister the provider
        llama_stack_client.providers.unregister(provider_id)

        # Verify it's no longer in the list
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id not in provider_ids

    def test_register_and_unregister_vector_store_provider(
        self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient
    ):
        """Test registering and unregistering a vector store provider."""
        provider_id = "test-dynamic-vector-store"

        # Clean up if exists
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register a new vector_io provider (using Faiss inline)
        response = llama_stack_client.providers.register(
            provider_id=provider_id,
            api="vector_io",
            provider_type="inline::faiss",
            config={
                "embedding_dimension": 768,
                "kvstore": {
                    "type": "sqlite",
                    "namespace": f"test_vector_store_{provider_id}",
                },
            },
        )

        # Verify registration
        assert response.provider.provider_id == provider_id
        assert response.provider.api == "vector_io"
        assert response.provider.provider_type == "inline::faiss"

        # Verify provider appears in list
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id in provider_ids

        # Unregister
        llama_stack_client.providers.unregister(provider_id)

        # Verify removal
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id not in provider_ids

    def test_update_provider_config(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test updating a provider's configuration."""
        provider_id = "test-update-config"

        # Clean up if exists
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register provider
        llama_stack_client.providers.register(
            provider_id=provider_id,
            api="inference",
            provider_type="remote::ollama",
            config={
                "url": "http://localhost:11434",
                "api_token": "old-token",
            },
        )

        # Update the configuration
        response = llama_stack_client.providers.update(
            provider_id=provider_id,
            config={
                "url": "http://localhost:11434",
                "api_token": "new-token",
            },
        )

        # Verify update
        assert response.provider.provider_id == provider_id
        assert response.provider.config["api_token"] == "new-token"

        # Clean up
        llama_stack_client.providers.unregister(provider_id)

    def test_update_provider_attributes(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test updating a provider's ABAC attributes."""
        provider_id = "test-update-attributes"

        # Clean up if exists
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register provider with initial attributes
        llama_stack_client.providers.register(
            provider_id=provider_id,
            api="inference",
            provider_type="remote::ollama",
            config={
                "url": "http://localhost:11434",
            },
            attributes={"team": ["team-a"]},
        )

        # Update attributes
        response = llama_stack_client.providers.update(
            provider_id=provider_id,
            attributes={"team": ["team-a", "team-b"], "environment": ["test"]},
        )

        # Verify attributes were updated
        assert response.provider.attributes["team"] == ["team-a", "team-b"]
        assert response.provider.attributes["environment"] == ["test"]

        # Clean up
        llama_stack_client.providers.unregister(provider_id)

    def test_test_provider_connection(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test the connection testing functionality."""
        provider_id = "test-connection-check"

        # Clean up if exists
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register provider
        llama_stack_client.providers.register(
            provider_id=provider_id,
            api="inference",
            provider_type="remote::ollama",
            config={
                "url": "http://localhost:11434",
            },
        )

        # Test the connection
        response = llama_stack_client.providers.test_connection(provider_id)

        # Verify response structure
        assert hasattr(response, "success")
        assert hasattr(response, "health")

        # Note: success may be True or False depending on whether Ollama is actually running
        # but the test should at least verify the API works

        # Clean up
        llama_stack_client.providers.unregister(provider_id)

    def test_register_duplicate_provider_fails(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test that registering a duplicate provider ID fails."""
        provider_id = "test-duplicate"

        # Clean up if exists
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register first provider
        llama_stack_client.providers.register(
            provider_id=provider_id,
            api="inference",
            provider_type="remote::ollama",
            config={"url": "http://localhost:11434"},
        )

        # Try to register with same ID - should fail
        with pytest.raises(Exception) as exc_info:
            llama_stack_client.providers.register(
                provider_id=provider_id,
                api="inference",
                provider_type="remote::ollama",
                config={"url": "http://localhost:11435"},
            )

        # Verify error message mentions the provider already exists
        assert "already exists" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()

        # Clean up
        llama_stack_client.providers.unregister(provider_id)

    def test_unregister_nonexistent_provider_fails(
        self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient
    ):
        """Test that unregistering a non-existent provider fails."""
        with pytest.raises(Exception) as exc_info:
            llama_stack_client.providers.unregister("nonexistent-provider-12345")

        # Verify error message mentions provider not found
        assert "not found" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_update_nonexistent_provider_fails(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test that updating a non-existent provider fails."""
        with pytest.raises(Exception) as exc_info:
            llama_stack_client.providers.update(
                provider_id="nonexistent-provider-12345",
                config={"url": "http://localhost:11434"},
            )

        # Verify error message mentions provider not found
        assert "not found" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_provider_lifecycle_with_inference(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test full lifecycle: register, use for inference (if Ollama available), update, unregister."""
        provider_id = "test-lifecycle-inference"

        # Clean up if exists
        try:
            llama_stack_client.providers.unregister(provider_id)
        except Exception:
            pass

        # Register provider
        response = llama_stack_client.providers.register(
            provider_id=provider_id,
            api="inference",
            provider_type="remote::ollama",
            config={
                "url": "http://localhost:11434",
            },
        )

        assert response.provider.status in ["connected", "initializing"]

        # Test connection
        conn_test = llama_stack_client.providers.test_connection(provider_id)
        assert hasattr(conn_test, "success")

        # Update configuration
        update_response = llama_stack_client.providers.update(
            provider_id=provider_id,
            config={
                "url": "http://localhost:11434",
                "api_token": "updated-token",
            },
        )
        assert update_response.provider.config["api_token"] == "updated-token"

        # Unregister
        llama_stack_client.providers.unregister(provider_id)

        # Verify it's gone
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id not in provider_ids

    def test_multiple_providers_same_type(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        """Test registering multiple providers of the same type with different IDs."""
        provider_id_1 = "test-multi-ollama-1"
        provider_id_2 = "test-multi-ollama-2"

        # Clean up if exists
        for pid in [provider_id_1, provider_id_2]:
            try:
                llama_stack_client.providers.unregister(pid)
            except Exception:
                pass

        # Register first provider
        response1 = llama_stack_client.providers.register(
            provider_id=provider_id_1,
            api="inference",
            provider_type="remote::ollama",
            config={"url": "http://localhost:11434"},
        )
        assert response1.provider.provider_id == provider_id_1

        # Register second provider with same type but different ID
        response2 = llama_stack_client.providers.register(
            provider_id=provider_id_2,
            api="inference",
            provider_type="remote::ollama",
            config={"url": "http://localhost:11434"},
        )
        assert response2.provider.provider_id == provider_id_2

        # Verify both are in the list
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id_1 in provider_ids
        assert provider_id_2 in provider_ids

        # Clean up both
        llama_stack_client.providers.unregister(provider_id_1)
        llama_stack_client.providers.unregister(provider_id_2)

        # Verify both are gone
        providers = llama_stack_client.providers.list()
        provider_ids = [p.provider_id for p in providers]
        assert provider_id_1 not in provider_ids
        assert provider_id_2 not in provider_ids
