# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for MLflow prompts provider configuration."""

import pytest
from pydantic import ValidationError

from llama_stack.providers.remote.prompts.mlflow.config import MLflowPromptsConfig


class TestMLflowPromptsConfig:
    """Tests for MLflowPromptsConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLflowPromptsConfig()

        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_registry_uri is None
        assert config.experiment_name == "llama-stack-prompts"
        assert config.use_metadata_tags is True
        assert config.timeout_seconds == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLflowPromptsConfig(
            mlflow_tracking_uri="http://mlflow.example.com:8080",
            mlflow_registry_uri="http://registry.example.com:8080",
            experiment_name="my-prompts",
            use_metadata_tags=False,
            timeout_seconds=60,
        )

        assert config.mlflow_tracking_uri == "http://mlflow.example.com:8080"
        assert config.mlflow_registry_uri == "http://registry.example.com:8080"
        assert config.experiment_name == "my-prompts"
        assert config.use_metadata_tags is False
        assert config.timeout_seconds == 60

    def test_databricks_uri(self):
        """Test Databricks URI configuration."""
        config = MLflowPromptsConfig(
            mlflow_tracking_uri="databricks",
            mlflow_registry_uri="databricks://profile",
        )

        assert config.mlflow_tracking_uri == "databricks"
        assert config.mlflow_registry_uri == "databricks://profile"

    def test_empty_tracking_uri(self):
        """Test validation rejects empty tracking URI."""
        with pytest.raises(ValidationError, match="mlflow_tracking_uri cannot be empty"):
            MLflowPromptsConfig(mlflow_tracking_uri="")

    def test_whitespace_tracking_uri(self):
        """Test validation rejects whitespace-only tracking URI."""
        with pytest.raises(ValidationError, match="mlflow_tracking_uri cannot be empty"):
            MLflowPromptsConfig(mlflow_tracking_uri="   ")

    def test_empty_experiment_name(self):
        """Test validation rejects empty experiment name."""
        with pytest.raises(ValidationError, match="experiment_name cannot be empty"):
            MLflowPromptsConfig(experiment_name="")

    def test_whitespace_experiment_name(self):
        """Test validation rejects whitespace-only experiment name."""
        with pytest.raises(ValidationError, match="experiment_name cannot be empty"):
            MLflowPromptsConfig(experiment_name="   ")

    def test_timeout_minimum_validation(self):
        """Test timeout must be >= 1."""
        with pytest.raises(ValidationError):
            MLflowPromptsConfig(timeout_seconds=0)

        with pytest.raises(ValidationError):
            MLflowPromptsConfig(timeout_seconds=-1)

    def test_timeout_maximum_validation(self):
        """Test timeout must be <= 300."""
        with pytest.raises(ValidationError):
            MLflowPromptsConfig(timeout_seconds=301)

        with pytest.raises(ValidationError):
            MLflowPromptsConfig(timeout_seconds=1000)

    def test_timeout_boundary_values(self):
        """Test timeout boundary values (1 and 300)."""
        config_min = MLflowPromptsConfig(timeout_seconds=1)
        assert config_min.timeout_seconds == 1

        config_max = MLflowPromptsConfig(timeout_seconds=300)
        assert config_max.timeout_seconds == 300

    def test_tracking_uri_strips_whitespace(self):
        """Test tracking URI whitespace is stripped."""
        config = MLflowPromptsConfig(mlflow_tracking_uri="  http://localhost:5000  ")
        assert config.mlflow_tracking_uri == "http://localhost:5000"

    def test_experiment_name_strips_whitespace(self):
        """Test experiment name whitespace is stripped."""
        config = MLflowPromptsConfig(experiment_name="  my-experiment  ")
        assert config.experiment_name == "my-experiment"

    def test_registry_uri_defaults_to_none(self):
        """Test registry URI defaults to None when not specified."""
        config = MLflowPromptsConfig()
        assert config.mlflow_registry_uri is None

    def test_use_metadata_tags_boolean(self):
        """Test use_metadata_tags accepts boolean values."""
        config_true = MLflowPromptsConfig(use_metadata_tags=True)
        assert config_true.use_metadata_tags is True

        config_false = MLflowPromptsConfig(use_metadata_tags=False)
        assert config_false.use_metadata_tags is False
