# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Post-Training API requests and responses.

This module defines the request and response models for the Post-Training API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack_api.common.content_types import URL
from llama_stack_api.common.job_types import JobStatus
from llama_stack_api.common.training_types import Checkpoint
from llama_stack_api.schema_utils import json_schema_type, register_schema


@json_schema_type
class OptimizerType(Enum):
    """Available optimizer algorithms for training.
    :cvar adam: Adaptive Moment Estimation optimizer
    :cvar adamw: AdamW optimizer with weight decay
    :cvar sgd: Stochastic Gradient Descent optimizer
    """

    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"


@json_schema_type
class DatasetFormat(Enum):
    """Format of the training dataset.
    :cvar instruct: Instruction-following format with prompt and completion
    :cvar dialog: Multi-turn conversation format with messages
    """

    instruct = "instruct"
    dialog = "dialog"


@json_schema_type
class DataConfig(BaseModel):
    """Configuration for training data and data loading.

    :param dataset_id: Unique identifier for the training dataset
    :param batch_size: Number of samples per training batch
    :param shuffle: Whether to shuffle the dataset during training
    :param data_format: Format of the dataset (instruct or dialog)
    :param validation_dataset_id: (Optional) Unique identifier for the validation dataset
    :param packed: (Optional) Whether to pack multiple samples into a single sequence for efficiency
    :param train_on_input: (Optional) Whether to compute loss on input tokens as well as output tokens
    """

    dataset_id: str
    batch_size: int
    shuffle: bool
    data_format: DatasetFormat
    validation_dataset_id: str | None = None
    packed: bool | None = False
    train_on_input: bool | None = False


@json_schema_type
class OptimizerConfig(BaseModel):
    """Configuration parameters for the optimization algorithm.

    :param optimizer_type: Type of optimizer to use (adam, adamw, or sgd)
    :param lr: Learning rate for the optimizer
    :param weight_decay: Weight decay coefficient for regularization
    :param num_warmup_steps: Number of steps for learning rate warmup
    """

    optimizer_type: OptimizerType
    lr: float
    weight_decay: float
    num_warmup_steps: int


@json_schema_type
class EfficiencyConfig(BaseModel):
    """Configuration for memory and compute efficiency optimizations.

    :param enable_activation_checkpointing: (Optional) Whether to use activation checkpointing to reduce memory usage
    :param enable_activation_offloading: (Optional) Whether to offload activations to CPU to save GPU memory
    :param memory_efficient_fsdp_wrap: (Optional) Whether to use memory-efficient FSDP wrapping
    :param fsdp_cpu_offload: (Optional) Whether to offload FSDP parameters to CPU
    """

    enable_activation_checkpointing: bool | None = False
    enable_activation_offloading: bool | None = False
    memory_efficient_fsdp_wrap: bool | None = False
    fsdp_cpu_offload: bool | None = False


@json_schema_type
class TrainingConfig(BaseModel):
    """Comprehensive configuration for the training process.

    :param n_epochs: Number of training epochs to run
    :param max_steps_per_epoch: Maximum number of steps to run per epoch
    :param gradient_accumulation_steps: Number of steps to accumulate gradients before updating
    :param max_validation_steps: (Optional) Maximum number of validation steps per epoch
    :param data_config: (Optional) Configuration for data loading and formatting
    :param optimizer_config: (Optional) Configuration for the optimization algorithm
    :param efficiency_config: (Optional) Configuration for memory and compute optimizations
    :param dtype: (Optional) Data type for model parameters (bf16, fp16, fp32)
    """

    n_epochs: int
    max_steps_per_epoch: int = 1
    gradient_accumulation_steps: int = 1
    max_validation_steps: int | None = 1
    data_config: DataConfig | None = None
    optimizer_config: OptimizerConfig | None = None
    efficiency_config: EfficiencyConfig | None = None
    dtype: str | None = "bf16"


@json_schema_type
class LoraFinetuningConfig(BaseModel):
    """Configuration for Low-Rank Adaptation (LoRA) fine-tuning.

    :param type: Algorithm type identifier, always "LoRA"
    :param lora_attn_modules: List of attention module names to apply LoRA to
    :param apply_lora_to_mlp: Whether to apply LoRA to MLP layers
    :param apply_lora_to_output: Whether to apply LoRA to output projection layers
    :param rank: Rank of the LoRA adaptation (lower rank = fewer parameters)
    :param alpha: LoRA scaling parameter that controls adaptation strength
    :param use_dora: (Optional) Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation)
    :param quantize_base: (Optional) Whether to quantize the base model weights
    """

    type: Literal["LoRA"] = "LoRA"
    lora_attn_modules: list[str]
    apply_lora_to_mlp: bool
    apply_lora_to_output: bool
    rank: int
    alpha: int
    use_dora: bool | None = False
    quantize_base: bool | None = False


@json_schema_type
class QATFinetuningConfig(BaseModel):
    """Configuration for Quantization-Aware Training (QAT) fine-tuning.

    :param type: Algorithm type identifier, always "QAT"
    :param quantizer_name: Name of the quantization algorithm to use
    :param group_size: Size of groups for grouped quantization
    """

    type: Literal["QAT"] = "QAT"
    quantizer_name: str
    group_size: int


AlgorithmConfig = Annotated[LoraFinetuningConfig | QATFinetuningConfig, Field(discriminator="type")]
register_schema(AlgorithmConfig, name="AlgorithmConfig")


@json_schema_type
class PostTrainingJobLogStream(BaseModel):
    """Stream of logs from a finetuning job.

    :param job_uuid: Unique identifier for the training job
    :param log_lines: List of log message strings from the training process
    """

    job_uuid: str
    log_lines: list[str]


@json_schema_type
class RLHFAlgorithm(Enum):
    """Available reinforcement learning from human feedback algorithms.
    :cvar dpo: Direct Preference Optimization algorithm
    """

    dpo = "dpo"


@json_schema_type
class DPOLossType(Enum):
    sigmoid = "sigmoid"
    hinge = "hinge"
    ipo = "ipo"
    kto_pair = "kto_pair"


@json_schema_type
class DPOAlignmentConfig(BaseModel):
    """Configuration for Direct Preference Optimization (DPO) alignment.

    :param beta: Temperature parameter for the DPO loss
    :param loss_type: The type of loss function to use for DPO
    """

    beta: float
    loss_type: DPOLossType = DPOLossType.sigmoid


@json_schema_type
class PostTrainingRLHFRequest(BaseModel):
    """Request to finetune a model using reinforcement learning from human feedback.

    :param job_uuid: Unique identifier for the training job
    :param finetuned_model: URL or path to the base model to fine-tune
    :param dataset_id: Unique identifier for the training dataset
    :param validation_dataset_id: Unique identifier for the validation dataset
    :param algorithm: RLHF algorithm to use for training
    :param algorithm_config: Configuration parameters for the RLHF algorithm
    :param optimizer_config: Configuration parameters for the optimization algorithm
    :param training_config: Configuration parameters for the training process
    :param hyperparam_search_config: Configuration for hyperparameter search
    :param logger_config: Configuration for training logging
    """

    job_uuid: str

    finetuned_model: URL

    dataset_id: str
    validation_dataset_id: str

    algorithm: RLHFAlgorithm
    algorithm_config: DPOAlignmentConfig

    optimizer_config: OptimizerConfig
    training_config: TrainingConfig

    # TODO: define these
    hyperparam_search_config: dict[str, Any]
    logger_config: dict[str, Any]


@json_schema_type
class PostTrainingJob(BaseModel):
    job_uuid: str


@json_schema_type
class PostTrainingJobStatusResponse(BaseModel):
    """Status of a finetuning job.

    :param job_uuid: Unique identifier for the training job
    :param status: Current status of the training job
    :param scheduled_at: (Optional) Timestamp when the job was scheduled
    :param started_at: (Optional) Timestamp when the job execution began
    :param completed_at: (Optional) Timestamp when the job finished, if completed
    :param resources_allocated: (Optional) Information about computational resources allocated to the job
    :param checkpoints: List of model checkpoints created during training
    """

    job_uuid: str
    status: JobStatus

    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    resources_allocated: dict[str, Any] | None = None

    checkpoints: list[Checkpoint] = Field(default_factory=list)


@json_schema_type
class ListPostTrainingJobsResponse(BaseModel):
    data: list[PostTrainingJob]


@json_schema_type
class PostTrainingJobArtifactsResponse(BaseModel):
    """Artifacts of a finetuning job.

    :param job_uuid: Unique identifier for the training job
    :param checkpoints: List of model checkpoints created during training
    """

    job_uuid: str
    checkpoints: list[Checkpoint] = Field(default_factory=list)

    # TODO(ashwin): metrics, evals


@json_schema_type
class SupervisedFineTuneRequest(BaseModel):
    """Request to run supervised fine-tuning of a model."""

    job_uuid: str = Field(..., description="The UUID of the job to create.")
    training_config: TrainingConfig = Field(..., description="The training configuration.")
    hyperparam_search_config: dict[str, Any] = Field(..., description="The hyperparam search configuration.")
    logger_config: dict[str, Any] = Field(..., description="The logger configuration.")
    model: str | None = Field(
        default=None,
        description="Model descriptor for training if not in provider config",
    )
    checkpoint_dir: str | None = Field(default=None, description="The directory to save checkpoint(s) to.")
    algorithm_config: AlgorithmConfig | None = Field(default=None, description="The algorithm configuration.")


@json_schema_type
class PreferenceOptimizeRequest(BaseModel):
    """Request to run preference optimization of a model."""

    job_uuid: str = Field(..., description="The UUID of the job to create.")
    finetuned_model: str = Field(..., description="The model to fine-tune.")
    algorithm_config: DPOAlignmentConfig = Field(..., description="The algorithm configuration.")
    training_config: TrainingConfig = Field(..., description="The training configuration.")
    hyperparam_search_config: dict[str, Any] = Field(..., description="The hyperparam search configuration.")
    logger_config: dict[str, Any] = Field(..., description="The logger configuration.")


@json_schema_type
class GetTrainingJobStatusRequest(BaseModel):
    """Request to get the status of a training job."""

    job_uuid: str = Field(..., description="The UUID of the job to get the status of.")


@json_schema_type
class CancelTrainingJobRequest(BaseModel):
    """Request to cancel a training job."""

    job_uuid: str = Field(..., description="The UUID of the job to cancel.")


@json_schema_type
class GetTrainingJobArtifactsRequest(BaseModel):
    """Request to get the artifacts of a training job."""

    job_uuid: str = Field(..., description="The UUID of the job to get the artifacts of.")
