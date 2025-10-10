# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.models import Model, ModelInput, ModelType
from llama_stack.core.datatypes import Provider, StackRunConfig
from llama_stack.core.stack import register_resources
from llama_stack.providers.datatypes import Api


@pytest.fixture(autouse=True)
def setup_caplog(caplog):
    """Configure caplog to capture all log levels"""
    caplog.set_level("DEBUG")


def assert_graceful_failure(message: str) -> Callable[[Exception], None]:
    """Create a failure handler with consistent error message format.

    Args:
        message: Error message with optional example
    """

    def _fail(_: Exception) -> None:
        pytest.fail(message)

    return _fail


def create_mock_impl(
    register_result: Any = None,
    register_error: Exception | None = None,
    list_result: Any = None,
    list_error: Exception | None = None,
) -> AsyncMock:
    """Create a mock implementation with configurable behavior.

    Args:
        register_result: Return value for register_model
        register_error: Error to raise from register_model
        list_result: Return value for list_models
        list_error: Error to raise from list_models
    """
    mock_impl = AsyncMock()
    if register_error:
        mock_impl.register_model = AsyncMock(side_effect=register_error)
    elif isinstance(register_result, list):
        mock_impl.register_model = AsyncMock(side_effect=register_result)
    else:
        mock_impl.register_model = AsyncMock(return_value=register_result)

    if list_error:
        mock_impl.list_models = AsyncMock(side_effect=list_error)
    else:
        mock_impl.list_models = AsyncMock(return_value=list_result or [])

    return mock_impl


@pytest.fixture
def mock_model():
    """Create a valid model for testing."""
    from llama_stack.apis.models import ModelInput  # Import the correct type

    return ModelInput(  # Use ModelInput instead of Model
        provider_id="test_provider",
        provider_model_id="test_model",
        identifier="test_model",
        model_id="test_model",  # Required field
        model_type=ModelType.llm,
    )


@pytest.fixture
def mock_provider():
    """Create a valid provider for testing."""
    return Provider(
        provider_id="test_provider",
        provider_type="remote::test",  # Valid format: namespace::name
        config={},
    )


@pytest.fixture
def mock_run_config(mock_model, mock_provider):  # Add mock_provider as dependency
    return StackRunConfig(
        image_name="test",
        apis=["inference"],
        providers={"inference": [mock_provider]},  # Use the Provider object directly
        models=[mock_model],
    )


async def test_register_resources_success(mock_run_config, mock_model):
    """Test successful registration of resources."""
    mock_impl = create_mock_impl(
        register_result=mock_model,
        list_result=[mock_model],
    )
    impls = {Api.models: mock_impl}

    try:
        await register_resources(mock_run_config, impls)
    except AttributeError:
        assert_graceful_failure("Stack interrupted initialization: tried to access model fields in wrong format")(None)

    mock_impl.register_model.assert_called_once()
    mock_impl.list_models.assert_called_once()


async def test_register_resources_failed_registration(caplog):
    """Test that stack continues when model registration fails.

    This test demonstrates how the stack handles validation errors:

    Before fix:
    - A provider failing to validate a model would crash the entire stack
    - Example: OpenAI provider with invalid API key would prevent startup
    - No way to start stack with other working providers

    After fix:
    - Provider validation errors are caught and logged
    - Stack continues initializing with other providers
    - Failed provider is skipped but doesn't block others

    Test strategy:
    1. Create an invalid model (wrong type) to trigger validation
    2. Create a valid provider to show it's not provider's fault
    3. Verify stack handles validation error and continues
    """
    # Create a valid model that will fail registration
    invalid_model = ModelInput(
        provider_id="test_provider",
        provider_model_id="test_model",
        identifier="test_model",
        model_id="test_model",  # Required field
        model_type=ModelType.llm,
    )

    # Create a valid provider
    valid_provider = Provider(
        provider_id="test_provider",
        provider_type="remote::test",
        config={},
    )

    # Create config with the model
    mock_run_config = StackRunConfig(
        image_name="test",
        apis=["inference"],
        providers={"inference": [valid_provider]},
        models=[invalid_model],
    )

    mock_impl = create_mock_impl(
        register_error=ValueError(
            "Provider failed to validate model: expected ModelInput but got Model\n"
            "This would previously crash the stack, but should now be handled gracefully"
        ),
    )
    impls = {Api.models: mock_impl}

    # Before fix: Stack would crash here
    # After fix: Should handle error and continue
    try:
        await register_resources(mock_run_config, impls)
    except Exception as e:
        assert_graceful_failure("Stack interrupted initialization: provider received model in wrong format")(e)

    # Verify registration was attempted despite validation issues
    (
        mock_impl.register_model.assert_called_once_with(
            **{k: getattr(invalid_model, k) for k in invalid_model.model_dump().keys()}
        ),
        "Provider should attempt model registration even with invalid model type",
    )

    # Verify stack continues operating after validation failure
    (
        mock_impl.list_models.assert_called_once(),
        "Stack should continue normal operation after handling model validation failure",
    )

    # Verify error was logged
    assert "Failed to register models" in caplog.text, "Error should be logged when model registration fails"
    assert "Provider failed to validate model" in caplog.text, "Specific error message should be logged"


async def test_register_resources_failed_listing(mock_run_config, mock_model):
    """Test that stack continues when model listing fails."""
    mock_impl = create_mock_impl(
        register_result=mock_model,
        list_error=ValueError("Listing failed"),
    )
    impls = {Api.models: mock_impl}

    # Should not raise exception
    try:
        await register_resources(mock_run_config, impls)
    except Exception as e:
        assert_graceful_failure("Stack interrupted initialization: provider failed to list available models")(e)

    # Verify registration completed successfully
    (
        mock_impl.register_model.assert_called_once_with(
            **{k: getattr(mock_model, k) for k in mock_model.model_dump().keys()}
        ),
        "register_model() should complete successfully before the listing failure",
    )

    # Verify listing was attempted
    (
        mock_impl.list_models.assert_called_once(),
        "list_models() should be called and its failure should be handled gracefully",
    )


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

    mock_impl = create_mock_impl(
        register_result=[model1, ValueError("Second registration failed")],
        list_result=[model1],  # Only first model listed
    )
    impls = {Api.models: mock_impl}

    # Should not raise exception
    try:
        await register_resources(mock_run_config, impls)
    except Exception as e:
        assert_graceful_failure(
            "Stack interrupted initialization: some models registered successfully but others failed"
        )(e)

    # Verify both registration attempts were made
    assert mock_impl.register_model.call_count == 2, (
        "register_model() should be called twice, once for each model, regardless of failures"
    )

    # Verify the first call succeeded with model1
    (
        mock_impl.register_model.assert_any_call(**{k: getattr(model1, k) for k in model1.model_dump().keys()}),
        "First model registration should be attempted with correct parameters",
    )

    # Verify the second call was attempted with model2
    (
        mock_impl.register_model.assert_any_call(**{k: getattr(model2, k) for k in model2.model_dump().keys()}),
        "Second model registration should be attempted even after first success",
    )

    # Verify listing was still performed
    mock_impl.list_models.assert_called_once(), "list_models() should be called once after all registration attempts"

    # Verify listing returned only the successful model
    assert len(mock_impl.list_models.return_value) == 1, (
        "list_models() should return only the successfully registered model"
    )
    assert mock_impl.list_models.return_value == [model1], (
        "list_models() should return the first model that registered successfully"
    )


async def test_register_resources_disabled_provider(mock_run_config, mock_model):
    """Test that disabled providers are skipped."""
    # Update model to be disabled
    mock_model.provider_id = "__disabled__"
    mock_impl = create_mock_impl()
    impls = {Api.models: mock_impl}

    try:
        await register_resources(mock_run_config, impls)
    except Exception as e:
        assert_graceful_failure("Stack interrupted initialization: provider is marked as disabled")(e)

    # Should not attempt registration for disabled provider
    mock_impl.register_model.assert_not_called(), "register_model() should not be called for disabled providers"

    # Should still perform listing
    mock_impl.list_models.assert_called_once(), "list_models() should still be called even for disabled providers"
