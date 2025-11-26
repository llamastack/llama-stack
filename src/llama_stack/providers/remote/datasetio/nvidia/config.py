# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import warnings
from typing import Any

from pydantic import BaseModel, Field


class NvidiaDatasetIOConfig(BaseModel):
    """Configuration for NVIDIA DatasetIO implementation."""

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY"),
        description="The NVIDIA API key.",
    )

    dataset_namespace: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASET_NAMESPACE", "default"),
        description="The NVIDIA dataset namespace.",
    )

    project_id: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_PROJECT_ID", "test-project"),
        description="The NVIDIA project ID.",
    )

    datasets_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASETS_URL", "http://nemo.test"),
        description="Base URL for the NeMo Dataset API",
    )

    # warning for default values
    def __post_init__(self):
        default_values = []
        if os.getenv("NVIDIA_PROJECT_ID") is None:
            default_values.append("project_id='test-project'")
        if os.getenv("NVIDIA_DATASET_NAMESPACE") is None:
            default_values.append("dataset_namespace='default'")
        if os.getenv("NVIDIA_DATASETS_URL") is None:
            default_values.append("datasets_url='http://nemo.test'")

        if default_values:
            warnings.warn(
                f"Using default values: {', '.join(default_values)}. \
                          Please set the environment variables to avoid this default behavior.",
                stacklevel=2,
            )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "api_key": "${env.NVIDIA_API_KEY:=}",
            "dataset_namespace": "${env.NVIDIA_DATASET_NAMESPACE:=default}",
            "project_id": "${env.NVIDIA_PROJECT_ID:=test-project}",
            "datasets_url": "${env.NVIDIA_DATASETS_URL:=http://nemo.test}",
        }
