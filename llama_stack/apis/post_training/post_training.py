# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.common.training_types import Checkpoint
from llama_stack.apis.version import LLAMA_STACK_API_V1ALPHA
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


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


class PostTraining(Protocol):
    @webmethod(route="/post-training/supervised-fine-tune", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str | None = Field(
            default=None,
            description="Model descriptor for training if not in provider config`",
        ),
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        """Run supervised fine-tuning of a model.

        :param job_uuid: The UUID of the job to create.
        :param training_config: The training configuration.
        :param hyperparam_search_config: The hyperparam search configuration.
        :param logger_config: The logger configuration.
        :param model: The model to fine-tune.
        :param checkpoint_dir: The directory to save checkpoint(s) to.
        :param algorithm_config: The algorithm configuration.
        :returns: A PostTrainingJob.
        """
        ...

    @webmethod(route="/post-training/preference-optimize", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        """Run preference optimization of a model.

        :param job_uuid: The UUID of the job to create.
        :param finetuned_model: The model to fine-tune.
        :param algorithm_config: The algorithm configuration.
        :param training_config: The training configuration.
        :param hyperparam_search_config: The hyperparam search configuration.
        :param logger_config: The logger configuration.
        :returns: A PostTrainingJob.
        """
        ...

    @webmethod(route="/post-training/jobs", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """Get all training jobs.

        :returns: A ListPostTrainingJobsResponse.
        """
        ...

    @webmethod(route="/post-training/job/status", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse:
        """Get the status of a training job.

        :param job_uuid: The UUID of the job to get the status of.
        :returns: A PostTrainingJobStatusResponse.
        """
        ...

    @webmethod(route="/post-training/job/cancel", method="POST", level=LLAMA_STACK_API_V1ALPHA)
    async def cancel_training_job(self, job_uuid: str) -> None:
        """Cancel a training job.

        :param job_uuid: The UUID of the job to cancel.
        """
        ...

    @webmethod(route="/post-training/job/artifacts", method="GET", level=LLAMA_STACK_API_V1ALPHA)
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse:
        """Get the artifacts of a training job.

        :param job_uuid: The UUID of the job to get the artifacts of.
        :returns: A PostTrainingJobArtifactsResponse.
        """
        ...
