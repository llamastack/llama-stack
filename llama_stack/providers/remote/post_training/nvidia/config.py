# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field

# TODO: add default values for all fields


class NvidiaPostTrainingConfig(BaseModel):
    """Configuration for NVIDIA Post Training implementation."""

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY"),
        description="The NVIDIA API key.",
    )

    dataset_namespace: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_DATASET_NAMESPACE", "default"),
        description="The NVIDIA dataset namespace.",
    )

    project_id: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_PROJECT_ID", "test-example-model@v1"),
        description="The NVIDIA project ID.",
    )

    # ToDO: validate this, add default value
    customizer_url: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_CUSTOMIZER_URL"),
        description="Base URL for the NeMo Customizer API",
    )

    timeout: int = Field(
        default=300,
        description="Timeout for the NVIDIA Post Training API",
    )

    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for the NVIDIA Post Training API",
    )

    # ToDo: validate this
    output_model_dir: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_OUTPUT_MODEL_DIR", "test-example-model@v1"),
        description="Directory to save the output model",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "api_key": "${env.NVIDIA_API_KEY:=}",
            "dataset_namespace": "${env.NVIDIA_DATASET_NAMESPACE:=default}",
            "project_id": "${env.NVIDIA_PROJECT_ID:=test-project}",
            "customizer_url": "${env.NVIDIA_CUSTOMIZER_URL:=http://nemo.test}",
        }


class SFTLoRADefaultConfig(BaseModel):
    """NVIDIA-specific training configuration with default values."""

    # ToDo: split into SFT and LoRA configs??

    # General training parameters
    n_epochs: int = 50

    # NeMo customizer specific parameters
    log_every_n_steps: int | None = None
    val_check_interval: float = 0.25
    sequence_packing_enabled: bool = False
    weight_decay: float = 0.01
    lr: float = 0.0001

    # SFT specific parameters
    hidden_dropout: float | None = None
    attention_dropout: float | None = None
    ffn_dropout: float | None = None

    # LoRA default parameters
    lora_adapter_dim: int = 8
    lora_adapter_dropout: float | None = None
    lora_alpha: int = 16

    # Data config
    batch_size: int = 8

    @classmethod
    def sample_config(cls) -> dict[str, Any]:
        """Return a sample configuration for NVIDIA training."""
        return {
            "n_epochs": 50,
            "log_every_n_steps": 10,
            "val_check_interval": 0.25,
            "sequence_packing_enabled": False,
            "weight_decay": 0.01,
            "hidden_dropout": 0.1,
            "attention_dropout": 0.1,
            "lora_adapter_dim": 8,
            "lora_alpha": 16,
            "data_config": {
                "dataset_id": "default",
                "batch_size": 8,
            },
            "optimizer_config": {
                "lr": 0.0001,
            },
        }
