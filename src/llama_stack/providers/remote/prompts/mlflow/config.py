# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Configuration for MLflow Prompt Registry provider.

This module defines the configuration schema for integrating Llama Stack
with MLflow's Prompt Registry for centralized prompt management and versioning.
"""

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator

from llama_stack_api import json_schema_type


class MLflowProviderDataValidator(BaseModel):
    """Validator for provider data from request headers.

    This allows users to override the MLflow API token per request via
    the x-llamastack-provider-data header:
        {"mlflow_api_token": "your-token"}
    """

    mlflow_api_token: str | None = Field(
        default=None,
        description="MLflow API token for authentication (overrides config)",
    )


@json_schema_type
class MLflowPromptsConfig(BaseModel):
    """Configuration for MLflow Prompt Registry provider.

    Credentials can be provided via:
    1. Per-request provider data header (preferred for security)
    2. Configuration auth_credential (fallback)
    3. Environment variables set by MLflow (MLFLOW_TRACKING_TOKEN, etc.)

    Attributes:
        mlflow_tracking_uri: MLflow tracking server URI (e.g., http://localhost:5000, databricks)
        mlflow_registry_uri: MLflow registry URI (optional, defaults to tracking_uri)
        experiment_name: MLflow experiment name for prompt storage
        auth_credential: MLflow API token for authentication (optional, can be overridden by provider data)
    """

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI (e.g., http://localhost:5000, databricks, databricks://profile)",
    )
    mlflow_registry_uri: str | None = Field(
        default=None,
        description="MLflow registry URI (defaults to tracking_uri if not specified)",
    )
    experiment_name: str = Field(
        default="llama-stack-prompts",
        description="MLflow experiment name for prompt storage and organization",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        description="MLflow API token for authentication. Can be overridden via provider data header.",
    )

    @classmethod
    def sample_run_config(cls, mlflow_api_token: str = "${env.MLFLOW_TRACKING_TOKEN:=}", **kwargs) -> dict[str, Any]:
        """Generate sample configuration with environment variable substitution.

        Args:
            mlflow_api_token: MLflow API token (defaults to MLFLOW_TRACKING_TOKEN env var)
            **kwargs: Additional configuration overrides

        Returns:
            Sample configuration dictionary
        """
        return {
            "mlflow_tracking_uri": kwargs.get("mlflow_tracking_uri", "http://localhost:5000"),
            "experiment_name": kwargs.get("experiment_name", "llama-stack-prompts"),
            "auth_credential": mlflow_api_token,
        }

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, v: str) -> str:
        """Validate tracking URI is not empty."""
        if not v or not v.strip():
            raise ValueError("mlflow_tracking_uri cannot be empty")
        return v.strip()

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, v: str) -> str:
        """Validate experiment name is not empty."""
        if not v or not v.strip():
            raise ValueError("experiment_name cannot be empty")
        return v.strip()
