# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for MLflow prompts provider configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from llama_stack.providers.remote.prompts.mlflow.config import (
    MLflowPromptsConfig,
    MLflowProviderDataValidator,
)


class TestMLflowPromptsConfig:
    """Tests for MLflowPromptsConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLflowPromptsConfig()

        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_registry_uri is None
        assert config.experiment_name == "llama-stack-prompts"
        assert config.auth_credential is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLflowPromptsConfig(
            mlflow_tracking_uri="http://mlflow.example.com:8080",
            mlflow_registry_uri="http://registry.example.com:8080",
            experiment_name="my-prompts",
            auth_credential=SecretStr("my-token"),
        )

        assert config.mlflow_tracking_uri == "http://mlflow.example.com:8080"
        assert config.mlflow_registry_uri == "http://registry.example.com:8080"
        assert config.experiment_name == "my-prompts"
        assert config.auth_credential.get_secret_value() == "my-token"

    def test_databricks_uri(self):
        """Test Databricks URI configuration."""
        config = MLflowPromptsConfig(
            mlflow_tracking_uri="databricks",
            mlflow_registry_uri="databricks://profile",
        )

        assert config.mlflow_tracking_uri == "databricks"
        assert config.mlflow_registry_uri == "databricks://profile"

    def test_tracking_uri_validation(self):
        """Test tracking URI validation."""
        # Empty string rejected
        with pytest.raises(ValidationError, match="mlflow_tracking_uri cannot be empty"):
            MLflowPromptsConfig(mlflow_tracking_uri="")

        # Whitespace-only rejected
        with pytest.raises(ValidationError, match="mlflow_tracking_uri cannot be empty"):
            MLflowPromptsConfig(mlflow_tracking_uri="   ")

        # Whitespace is stripped
        config = MLflowPromptsConfig(mlflow_tracking_uri="  http://localhost:5000  ")
        assert config.mlflow_tracking_uri == "http://localhost:5000"

    def test_experiment_name_validation(self):
        """Test experiment name validation."""
        # Empty string rejected
        with pytest.raises(ValidationError, match="experiment_name cannot be empty"):
            MLflowPromptsConfig(experiment_name="")

        # Whitespace-only rejected
        with pytest.raises(ValidationError, match="experiment_name cannot be empty"):
            MLflowPromptsConfig(experiment_name="   ")

        # Whitespace is stripped
        config = MLflowPromptsConfig(experiment_name="  my-experiment  ")
        assert config.experiment_name == "my-experiment"

    def test_sample_run_config(self):
        """Test sample_run_config generates valid configuration."""
        # Default environment variable
        sample = MLflowPromptsConfig.sample_run_config()
        assert sample["mlflow_tracking_uri"] == "http://localhost:5000"
        assert sample["experiment_name"] == "llama-stack-prompts"
        assert sample["auth_credential"] == "${env.MLFLOW_TRACKING_TOKEN:=}"

        # Custom values
        sample = MLflowPromptsConfig.sample_run_config(
            mlflow_api_token="test-token",
            mlflow_tracking_uri="http://custom:5000",
        )
        assert sample["mlflow_tracking_uri"] == "http://custom:5000"
        assert sample["auth_credential"] == "test-token"


class TestMLflowProviderDataValidator:
    """Tests for MLflowProviderDataValidator."""

    def test_provider_data_validator(self):
        """Test provider data validator with and without token."""
        # With token
        validator = MLflowProviderDataValidator(mlflow_api_token="test-token-123")
        assert validator.mlflow_api_token == "test-token-123"

        # Without token
        validator = MLflowProviderDataValidator()
        assert validator.mlflow_api_token is None

        # From dictionary
        data = {"mlflow_api_token": "secret-token"}
        validator = MLflowProviderDataValidator(**data)
        assert validator.mlflow_api_token == "secret-token"
