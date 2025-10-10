# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.models import Model, ModelType
from llama_stack.core.datatypes import Provider, StackRunConfig
from llama_stack.core.stack import Stack, register_resources
from llama_stack.providers.datatypes import Api


@pytest.fixture
def mock_model():
    return Model(
        provider_id="test_provider",
        provider_model_id="test_model",
        identifier="test_model",
        model_type=ModelType.llm,
    )


@pytest.fixture
def mock_provider():
    return Provider(
        provider_id="test_provider",
        provider_type="test_type",
        config={},
    )


@pytest.fixture
def mock_run_config(mock_model):
    return StackRunConfig(
        image_name="test",
        apis=["inference"],
        providers={"inference": [mock_provider]},
        models=[mock_model],
    )


async def test_register_resources_success(mock_run_config, mock_model):
    """Test successful registration of resources."""
    mock_impl = AsyncMock()
    mock_impl.register_model = AsyncMock(return_value=mock_model)
    mock_impl.list_models = AsyncMock(return_value=[mock_model])

    impls = {Api.models: mock_impl}

    await register_resources(mock_run_config, impls)

    mock_impl.register_model.assert_called_once()
    mock_impl.list_models.assert_called_once()


async def test_register_resources_failed_registration(mock_run_config, mock_model):
    """Test that stack continues when model registration fails."""
    mock_impl = AsyncMock()
    mock_impl.register_model = AsyncMock(side_effect=ValueError("Registration failed"))
    mock_impl.list_models = AsyncMock(return_value=[])

    impls = {Api.models: mock_impl}

    # Should not raise exception
    await register_resources(mock_run_config, impls)

    mock_impl.register_model.assert_called_once()
    mock_impl.list_models.assert_called_once()


async def test_register_resources_failed_listing(mock_run_config, mock_model):
    """Test that stack continues when model listing fails."""
    mock_impl = AsyncMock()
    mock_impl.register_model = AsyncMock(return_value=mock_model)
    mock_impl.list_models = AsyncMock(side_effect=ValueError("Listing failed"))

    impls = {Api.models: mock_impl}

    # Should not raise exception
    await register_resources(mock_run_config, impls)

    mock_impl.register_model.assert_called_once()
    mock_impl.list_models.assert_called_once()


async def test_register_resources_mixed_success(mock_run_config):
    """Test mixed success/failure scenario with multiple models."""
    # Create two models
    model1 = Model(
        provider_id="test_provider",
        provider_model_id="model1",
        identifier="model1",
        model_type=ModelType.llm,
    )
    model2 = Model(
        provider_id="test_provider",
        provider_model_id="model2",
        identifier="model2",
        model_type=ModelType.llm,
    )

    # Update run config to include both models
    mock_run_config.models = [model1, model2]

    mock_impl = AsyncMock()
    # Make first registration succeed, second fail
    mock_impl.register_model = AsyncMock(side_effect=[model1, ValueError("Second registration failed")])
    mock_impl.list_models = AsyncMock(return_value=[model1])  # Only first model listed

    impls = {Api.models: mock_impl}

    # Should not raise exception
    await register_resources(mock_run_config, impls)

    assert mock_impl.register_model.call_count == 2
    mock_impl.list_models.assert_called_once()


async def test_register_resources_disabled_provider(mock_run_config, mock_model):
    """Test that disabled providers are skipped."""
    # Update model to be disabled
    mock_model.provider_id = "__disabled__"
    mock_impl = AsyncMock()

    impls = {Api.models: mock_impl}

    await register_resources(mock_run_config, impls)

    # Should not attempt registration for disabled provider
    mock_impl.register_model.assert_not_called()
    mock_impl.list_models.assert_called_once()


class MockFailingProvider:
    """A mock provider that fails registration but allows initialization"""

    def __init__(self, *args, **kwargs):
        self.initialize_called = False
        self.shutdown_called = False

    async def initialize(self):
        self.initialize_called = True

    async def shutdown(self):
        self.shutdown_called = True

    async def register_model(self, *args, **kwargs):
        raise ValueError("Mock registration failure")

    async def list_models(self):
        return []  # Return empty list to simulate no models registered


async def test_stack_initialization_with_failed_registration():
    """Test full stack initialization with failed model registration using a mock provider."""
    mock_model = Model(
        provider_id="mock_failing",
        provider_model_id="test_model",
        identifier="test_model",
        model_type=ModelType.llm,
    )

    mock_run_config = StackRunConfig(
        image_name="test",
        apis=["inference"],
        providers={
            "inference": [
                Provider(
                    provider_id="mock_failing",
                    provider_type="mock::failing",
                    config={},  # No need for real config in mock
                )
            ]
        },
        models=[mock_model],
    )

    # Create a mock provider registry that returns our failing provider
    mock_registry = {
        Api.inference: {
            "mock::failing": MagicMock(
                provider_class=MockFailingProvider,
                config_class=MagicMock(),
            )
        }
    }

    stack = Stack(mock_run_config, provider_registry=mock_registry)

    # Should not raise exception during initialization
    await stack.initialize()

    # Stack should still be initialized
    assert stack.impls is not None

    # Verify the provider was properly initialized
    inference_impl = stack.impls.get(Api.inference)
    assert inference_impl is not None
    assert inference_impl.initialize_called

    # Clean up
    await stack.shutdown()
    assert inference_impl.shutdown_called
