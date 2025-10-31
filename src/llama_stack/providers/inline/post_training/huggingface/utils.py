# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import psutil
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class HFAutoModel(Protocol):
    """Protocol describing HuggingFace AutoModel interface.

    This protocol defines the common interface for HuggingFace AutoModelForCausalLM
    and similar models, providing type safety without requiring type stubs.
    """

    config: PretrainedConfig
    device: torch.device

    def to(self, device: torch.device) -> "HFAutoModel": ...
    def save_pretrained(self, save_directory: str | Path) -> None: ...


from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.post_training import Checkpoint, TrainingConfig
from llama_stack.log import get_logger

from .config import HuggingFacePostTrainingConfig

logger = get_logger(name=__name__, category="post_training")


def setup_environment():
    """Setup common environment variables for training."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
    os.environ["MKL_NUM_THREADS"] = "1"


def bytes_to_gb(to_convert: int) -> str:
    """Converts memory stats to GB and formats to 2 decimal places.
    Args:
        to_convert: Memory value in bytes
    Returns:
        str: Memory value in GB formatted to 2 decimal places
    """
    return f"{(to_convert / (1024**3)):.2f}"


def get_memory_stats(device: torch.device) -> dict[str, Any]:
    """Get memory statistics for the given device."""
    stats = {
        "system_memory": {
            "total": bytes_to_gb(psutil.virtual_memory().total),
            "available": bytes_to_gb(psutil.virtual_memory().available),
            "used": bytes_to_gb(psutil.virtual_memory().used),
            "percent": psutil.virtual_memory().percent,
        }
    }

    if device.type == "cuda":
        stats["device_memory"] = {
            "allocated": bytes_to_gb(torch.cuda.memory_allocated(device)),
            "reserved": bytes_to_gb(torch.cuda.memory_reserved(device)),
            "max_allocated": bytes_to_gb(torch.cuda.max_memory_allocated(device)),
        }
    elif device.type == "mps":
        # MPS doesn't provide direct memory stats, but we can track system memory
        stats["device_memory"] = {
            "note": "MPS memory stats not directly available",
            "system_memory_used": bytes_to_gb(psutil.virtual_memory().used),
        }
    elif device.type == "cpu":
        # For CPU, we track process memory usage
        process = psutil.Process()
        stats["device_memory"] = {
            "process_rss": bytes_to_gb(process.memory_info().rss),
            "process_vms": bytes_to_gb(process.memory_info().vms),
            "process_percent": process.memory_percent(),
        }

    return stats


def setup_torch_device(device_str: str) -> torch.device:
    """Initialize and validate a PyTorch device.
    This function handles device initialization and validation for different device types:
    - CUDA: Validates CUDA availability and handles device selection
    - MPS: Validates MPS availability for Apple Silicon
    - CPU: Basic validation
    - HPU: Raises error as it's not supported
    Args:
        device_str: String specifying the device ('cuda', 'cpu', 'mps')
    Returns:
        torch.device: The initialized and validated device
    Raises:
        RuntimeError: If device initialization fails or device is not supported
    """
    try:
        device = torch.device(device_str)
    except RuntimeError as e:
        raise RuntimeError(f"Error getting Torch Device {str(e)}") from e

    # Validate device capabilities
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{device.type}: Torch has no CUDA/ROCm support or could not detect a compatible device."
            )
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"{device.type}: Torch has no MPS support or could not detect a compatible device.")
    elif device.type == "hpu":
        raise RuntimeError(f"{device.type}: training does not support Intel Gaudi.")

    return device


async def load_rows_from_dataset(datasetio_api: DatasetIO, dataset_id: str) -> list[dict[str, Any]]:
    """Load dataset from llama stack dataset provider"""
    try:
        all_rows = await datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=-1,
        )
        if not isinstance(all_rows.data, list):
            raise RuntimeError("Expected dataset data to be a list")
        return all_rows.data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}") from e


def load_model(
    model: str,
    device: torch.device,
    provider_config: HuggingFacePostTrainingConfig,
) -> HFAutoModel:
    """Load and initialize the model for training.
    Args:
        model: The model identifier to load
        device: The device to load the model onto
        provider_config: Provider-specific configuration
    Returns:
        The loaded and initialized model
    Raises:
        RuntimeError: If model loading fails
    """
    from typing import cast

    logger.info("Loading the base model")
    try:
        model_config = AutoConfig.from_pretrained(model, **provider_config.model_specific_config)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto" if device.type != "cpu" else "float32",
            quantization_config=None,
            config=model_config,
            **provider_config.model_specific_config,
        )
        # Always move model to specified device
        model_obj = model_obj.to(device)  # type: ignore[arg-type]
        logger.info(f"Model loaded and moved to device: {model_obj.device}")
        # Cast to HFAutoModel protocol - transformers models satisfy this interface
        return cast(HFAutoModel, model_obj)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}") from e


def split_dataset(ds: Dataset) -> tuple[Dataset, Dataset]:
    """Split dataset into train and validation sets.
    Args:
        ds: Dataset to split
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    logger.info("Splitting dataset into train and validation sets")
    train_val_split = ds.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    logger.info(f"Split dataset into {len(train_dataset)} training and {len(eval_dataset)} validation examples")
    return train_dataset, eval_dataset


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def calculate_training_steps(steps_per_epoch: int, config: TrainingConfig) -> dict[str, int]:
    """Calculate training steps and logging configuration.
    Args:
        steps_per_epoch: Number of training steps per epoch
        config: Training configuration
    Returns:
        dict: Dictionary with calculated step values
    """
    total_steps = steps_per_epoch * config.n_epochs
    max_steps = min(config.max_steps_per_epoch, total_steps)
    logging_steps = max(1, steps_per_epoch // 50)  # Log 50 times per epoch

    logger.info("Training configuration:")
    logger.info(f"- Steps per epoch: {steps_per_epoch}")
    logger.info(f"- Total steps: {total_steps}")
    logger.info(f"- Max steps: {max_steps}")
    logger.info(f"- Logging steps: {logging_steps}")

    return {"total_steps": total_steps, "max_steps": max_steps, "logging_steps": logging_steps}


def get_save_strategy(output_dir_path: Path | None) -> tuple[str, str]:
    """Get save and evaluation strategy based on output directory.
    Args:
        output_dir_path: Optional path to save the model
    Returns:
        tuple: (save_strategy, eval_strategy)
    """
    if output_dir_path:
        logger.info(f"Will save checkpoints to {output_dir_path}")
        return "epoch", "epoch"
    return "no", "no"


def create_checkpoints(
    output_dir_path: Path, job_uuid: str, model: str, config: TrainingConfig, final_model_name: str
) -> list[Checkpoint]:
    """Create checkpoint objects from training output.
    Args:
        output_dir_path: Path to the training output directory
        job_uuid: Unique identifier for the training job
        model: Model identifier
        config: Training configuration
        final_model_name: Name of the final model directory ("merged_model" for SFT, "dpo_model" for DPO)
    Returns:
        List of Checkpoint objects
    """
    checkpoints = []

    # Add checkpoint directories
    checkpoint_dirs = sorted(
        [d for d in output_dir_path.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[1]),
    )

    for epoch_number, checkpoint_dir in enumerate(checkpoint_dirs, start=1):
        created_time = datetime.fromtimestamp(os.path.getctime(checkpoint_dir), tz=UTC)
        checkpoint = Checkpoint(
            identifier=checkpoint_dir.name,
            created_at=created_time,
            epoch=epoch_number,
            post_training_job_id=job_uuid,
            path=str(checkpoint_dir),
        )
        checkpoints.append(checkpoint)

    # Add final model
    final_model_path = output_dir_path / final_model_name
    if final_model_path.exists():
        training_type = "sft" if final_model_name == "merged_model" else "dpo"
        checkpoint = Checkpoint(
            identifier=f"{model}-{training_type}-{config.n_epochs}",
            created_at=datetime.now(UTC),
            epoch=config.n_epochs,
            post_training_job_id=job_uuid,
            path=str(final_model_path),
        )
        checkpoints.append(checkpoint)

    return checkpoints
