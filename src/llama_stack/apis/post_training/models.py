# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.common.training_types import Checkpoint
from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class OptimizerType(Enum):
    """Available optimizer algorithms for training."""

    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"


@json_schema_type
class DatasetFormat(Enum):
    """Format of the training dataset."""

    instruct = "instruct"
    dialog = "dialog"


@json_schema_type
class DataConfig(BaseModel):
    """Configuration for training data and data loading."""

    dataset_id: str
    batch_size: int
    shuffle: bool
    data_format: DatasetFormat
    validation_dataset_id: str | None = None
    packed: bool | None = False
    train_on_input: bool | None = False


@json_schema_type
class OptimizerConfig(BaseModel):
    """Configuration parameters for the optimization algorithm."""

    optimizer_type: OptimizerType
    lr: float
    weight_decay: float
    num_warmup_steps: int


@json_schema_type
class EfficiencyConfig(BaseModel):
    """Configuration for memory and compute efficiency optimizations."""

    enable_activation_checkpointing: bool | None = False
    enable_activation_offloading: bool | None = False
    memory_efficient_fsdp_wrap: bool | None = False
    fsdp_cpu_offload: bool | None = False


@json_schema_type
class TrainingConfig(BaseModel):
    """Comprehensive configuration for the training process."""

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
    """Configuration for Low-Rank Adaptation (LoRA) fine-tuning."""

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
    """Configuration for Quantization-Aware Training (QAT) fine-tuning."""

    type: Literal["QAT"] = "QAT"
    quantizer_name: str
    group_size: int


AlgorithmConfig = Annotated[LoraFinetuningConfig | QATFinetuningConfig, Field(discriminator="type")]
register_schema(AlgorithmConfig, name="AlgorithmConfig")


@json_schema_type
class PostTrainingJobLogStream(BaseModel):
    """Stream of logs from a finetuning job."""

    job_uuid: str
    log_lines: list[str]


@json_schema_type
class RLHFAlgorithm(Enum):
    """Available reinforcement learning from human feedback algorithms."""

    dpo = "dpo"


@json_schema_type
class DPOLossType(Enum):
    sigmoid = "sigmoid"
    hinge = "hinge"
    ipo = "ipo"
    kto_pair = "kto_pair"


@json_schema_type
class DPOAlignmentConfig(BaseModel):
    """Configuration for Direct Preference Optimization (DPO) alignment."""

    beta: float
    loss_type: DPOLossType = DPOLossType.sigmoid


@json_schema_type
class PostTrainingRLHFRequest(BaseModel):
    """Request to finetune a model using reinforcement learning from human feedback."""

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
    job_uuid: str = Field(..., description="The UUID of the job")


@json_schema_type
class PostTrainingJobStatusResponse(BaseModel):
    """Status of a finetuning job."""

    job_uuid: str
    status: JobStatus

    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    resources_allocated: dict[str, Any] | None = None

    checkpoints: list[Checkpoint] = Field(default_factory=list)


class ListPostTrainingJobsResponse(BaseModel):
    data: list[PostTrainingJob] = Field(..., description="The list of training jobs")


@json_schema_type
class PostTrainingJobArtifactsResponse(BaseModel):
    """Artifacts of a finetuning job."""

    job_uuid: str = Field(..., description="The UUID of the job")
    checkpoints: list[Checkpoint] = Field(default_factory=list)

    # TODO(ashwin): metrics, evals


@json_schema_type
class SupervisedFineTuneRequest(BaseModel):
    """Request to run supervised fine-tuning of a model."""

    job_uuid: str = Field(..., description="The UUID of the job to create")
    training_config: TrainingConfig = Field(..., description="The training configuration")
    hyperparam_search_config: dict[str, Any] = Field(..., description="The hyperparam search configuration")
    logger_config: dict[str, Any] = Field(..., description="The logger configuration")
    model: str | None = Field(
        default=None,
        description="Model descriptor for training if not in provider config`",
    )
    checkpoint_dir: str | None = Field(default=None, description="The directory to save checkpoint(s) to")
    algorithm_config: AlgorithmConfig | None = Field(default=None, description="The algorithm configuration")


@json_schema_type
class PreferenceOptimizeRequest(BaseModel):
    """Request to run preference optimization of a model."""

    job_uuid: str = Field(..., description="The UUID of the job to create")
    finetuned_model: str = Field(..., description="The model to fine-tune")
    algorithm_config: DPOAlignmentConfig = Field(..., description="The algorithm configuration")
    training_config: TrainingConfig = Field(..., description="The training configuration")
    hyperparam_search_config: dict[str, Any] = Field(..., description="The hyperparam search configuration")
    logger_config: dict[str, Any] = Field(..., description="The logger configuration")
