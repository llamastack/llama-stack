# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Configuration for MLflow Prompt Registry provider.

This module defines the configuration schema for integrating Llama Stack
with MLflow's Prompt Registry for centralized prompt management and versioning.
"""

from pydantic import BaseModel, Field, field_validator


class MLflowPromptsConfig(BaseModel):
    """Configuration for MLflow Prompt Registry provider.

    Attributes:
        mlflow_tracking_uri: MLflow tracking server URI (e.g., http://localhost:5000, databricks)
        mlflow_registry_uri: MLflow registry URI (optional, defaults to tracking_uri)
        experiment_name: MLflow experiment name for prompt storage
        use_metadata_tags: Store Llama Stack metadata in MLflow tags (default: True)
        timeout_seconds: Timeout for MLflow API calls in seconds (default: 30)
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
    use_metadata_tags: bool = Field(
        default=True,
        description="Store Llama Stack metadata (prompt_id, variables) in MLflow tags",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for MLflow API calls in seconds (1-300)",
    )

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
