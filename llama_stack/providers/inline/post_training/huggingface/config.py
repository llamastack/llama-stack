# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel


class HuggingFacePostTrainingConfig(BaseModel):
    # Device to run training on (cuda, cpu, mps)
    device: str = "cuda"

    # Distributed training backend if using multiple devices
    # fsdp: Fully Sharded Data Parallel
    # deepspeed: DeepSpeed ZeRO optimization
    distributed_backend: Literal["fsdp", "deepspeed"] | None = None

    # Format for saving model checkpoints
    # full_state: Save complete model state
    # huggingface: Save in HuggingFace format (recommended for compatibility)
    checkpoint_format: Literal["full_state", "huggingface"] | None = "huggingface"

    # Template for formatting chat inputs and outputs
    # Used to structure the conversation format for training
    chat_template: str = "<|user|>\n{input}\n<|assistant|>\n{output}"

    # Model-specific configuration parameters
    # trust_remote_code: Allow execution of custom model code
    # attn_implementation: Use SDPA (Scaled Dot Product Attention) for better performance
    model_specific_config: dict = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    # Maximum sequence length for training
    # Set to 2048 as this is the maximum that works reliably on MPS (Apple Silicon)
    # Longer sequences may cause memory issues on MPS devices
    max_seq_length: int = 2048

    # Enable gradient checkpointing to reduce memory usage
    # Trades computation for memory by recomputing activations
    gradient_checkpointing: bool = False

    # Maximum number of checkpoints to keep
    # Older checkpoints are deleted when this limit is reached
    save_total_limit: int = 3

    # Number of training steps between logging updates
    logging_steps: int = 10

    # Ratio of training steps used for learning rate warmup
    # Helps stabilize early training
    warmup_ratio: float = 0.1

    # L2 regularization coefficient
    # Helps prevent overfitting
    weight_decay: float = 0.01

    # Number of worker processes for data loading
    # Higher values can improve data loading speed but increase memory usage
    dataloader_num_workers: int = 4

    # Whether to pin memory in data loader
    # Can improve data transfer speed to GPU but uses more memory
    dataloader_pin_memory: bool = True

    # DPO-specific parameters
    dpo_beta: float = 0.1
    use_reference_model: bool = True
    dpo_loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid"
    dpo_output_dir: str

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "checkpoint_format": "huggingface",
            "distributed_backend": None,
            "device": "cpu",
            "dpo_output_dir": __distro_dir__ + "/dpo_output",
        }
