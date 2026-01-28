# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Fine-Tuning API requests and responses.

This module defines OpenAI-compatible models for the Fine-Tuning API,
matching the structure and behavior of OpenAI's fine-tuning endpoints.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from llama_stack_api.schema_utils import json_schema_type, nullable_openai_style, remove_null_from_anyof


@json_schema_type
class FineTuningJobStatus(str, Enum):
    """Status of a fine-tuning job.

    Matches OpenAI's fine-tuning job status values.

    :cvar VALIDATING_FILES: Job is validating input files
    :cvar QUEUED: Job is queued for execution
    :cvar RUNNING: Job is currently running
    :cvar SUCCEEDED: Job completed successfully
    :cvar FAILED: Job failed during execution
    :cvar CANCELLED: Job was cancelled by user
    """

    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@json_schema_type
class FineTuningError(BaseModel):
    """Error information for failed fine-tuning jobs.

    :param code: Machine-readable error code
    :param message: Human-readable error message
    :param param: Parameter that caused the error (null if not parameter-specific)
    """

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    param: str | None = Field(
        ...,
        description="Parameter that caused the error (null if not parameter-specific)",
        json_schema_extra=nullable_openai_style,
    )


@json_schema_type
class FineTuneSupervisedHyperparameters(BaseModel):
    """Hyperparameters for supervised fine-tuning.

    :param n_epochs: Number of training epochs
    :param batch_size: Training batch size
    :param learning_rate_multiplier: Learning rate multiplier
    """

    n_epochs: int | None = Field(None, description="Number of training epochs", json_schema_extra=nullable_openai_style)
    batch_size: int | None = Field(None, description="Training batch size", json_schema_extra=nullable_openai_style)
    learning_rate_multiplier: float | None = Field(
        None, description="Learning rate multiplier", json_schema_extra=nullable_openai_style
    )


@json_schema_type
class FineTuneDPOHyperparameters(BaseModel):
    """Hyperparameters for DPO fine-tuning.

    :param n_epochs: Number of training epochs
    :param batch_size: Training batch size
    :param learning_rate_multiplier: Learning rate multiplier
    :param beta: DPO beta parameter
    """

    n_epochs: int | None = Field(None, description="Number of training epochs", json_schema_extra=nullable_openai_style)
    batch_size: int | None = Field(None, description="Training batch size", json_schema_extra=nullable_openai_style)
    learning_rate_multiplier: float | None = Field(
        None, description="Learning rate multiplier", json_schema_extra=nullable_openai_style
    )
    beta: float | None = Field(None, description="DPO beta parameter", json_schema_extra=nullable_openai_style)


@json_schema_type
class FineTuneSupervisedMethod(BaseModel):
    """Configuration for supervised fine-tuning method.

    :param hyperparameters: Hyperparameters for supervised training
    """

    hyperparameters: FineTuneSupervisedHyperparameters | None = Field(None, json_schema_extra=nullable_openai_style)


@json_schema_type
class FineTuneDPOMethod(BaseModel):
    """Configuration for DPO fine-tuning method.

    :param hyperparameters: Hyperparameters for DPO training
    """

    hyperparameters: FineTuneDPOHyperparameters | None = Field(None, json_schema_extra=nullable_openai_style)


@json_schema_type
class FineTuneReinforcementMethod(BaseModel):
    """Configuration for reinforcement fine-tuning method.

    :param hyperparameters: Hyperparameters for reinforcement training
    """

    hyperparameters: FineTuneDPOHyperparameters | None = Field(None, json_schema_extra=nullable_openai_style)


@json_schema_type
class FineTuneMethod(BaseModel):
    """The method used for fine-tuning.

    Matches OpenAI's structure where type is discriminator and
    method-specific config is in optional fields.

    :param type: Type of fine-tuning method
    :param supervised: Configuration for supervised method (only if type=supervised)
    :param dpo: Configuration for DPO method (only if type=dpo)
    :param reinforcement: Configuration for reinforcement method (only if type=reinforcement)
    """

    type: Literal["supervised", "dpo", "reinforcement"] = Field(..., description="Fine-tuning method type")
    supervised: FineTuneSupervisedMethod | None = Field(
        None, description="Supervised fine-tuning configuration", json_schema_extra=nullable_openai_style
    )
    dpo: FineTuneDPOMethod | None = Field(
        None, description="DPO fine-tuning configuration", json_schema_extra=nullable_openai_style
    )
    reinforcement: FineTuneReinforcementMethod | None = Field(
        None, description="Reinforcement fine-tuning configuration", json_schema_extra=nullable_openai_style
    )


@json_schema_type
class FineTuningJobEvent(BaseModel):
    """A log event from a fine-tuning job.

    :param id: Event identifier
    :param created_at: Unix timestamp of event
    :param level: Log level (info, warn, warning, error)
    :param message: Event message
    :param object: Object type (always 'fine_tuning.job.event')
    """

    id: str = Field(..., description="Event identifier")
    created_at: int = Field(..., description="Unix timestamp")
    level: Literal["info", "warn", "warning", "error"] = Field(..., description="Log level")
    message: str = Field(..., description="Event message")
    object: Literal["fine_tuning.job.event"] = Field(default="fine_tuning.job.event", description="Object type")


@json_schema_type
class FineTuningJobHyperparameters(BaseModel):
    """Hyperparameters object matching OpenAI's structure.

    :param n_epochs: Number of training epochs (default: "auto")
    :param batch_size: Training batch size (default: "auto")
    :param learning_rate_multiplier: Learning rate multiplier (default: "auto")
    """

    n_epochs: int | Literal["auto"] = Field(default="auto", description="Number of training epochs")
    batch_size: int | Literal["auto"] = Field(default="auto", description="Training batch size")
    learning_rate_multiplier: float | Literal["auto"] = Field(default="auto", description="Learning rate multiplier")


@json_schema_type
class FineTuningJob(BaseModel):
    """A fine-tuning job.

    Matches OpenAI's FineTuningJob structure with Llama Stack extensions.

    :param id: Job identifier
    :param object: Object type (always 'fine_tuning.job')
    :param created_at: Unix timestamp when job was created
    :param finished_at: Unix timestamp when job finished (null if running)
    :param model: Base model being fine-tuned
    :param fine_tuned_model: Resulting fine-tuned model identifier (null if not complete)
    :param status: Current job status
    :param training_file: Training dataset identifier
    :param validation_file: Validation dataset identifier (optional)
    :param hyperparameters: Hyperparameters used for training
    :param trained_tokens: Total tokens processed (null if running)
    :param error: Error details if job failed
    :param seed: Random seed used
    :param estimated_finish: Estimated completion time (null if not available)
    :param method: Fine-tuning method configuration
    :param result_files: List of result file identifiers
    :param integrations: List of integration configurations (nullable in OpenAI)
    :param metadata: Custom metadata (OpenAI field)
    :param organization_id: Organization identifier (OpenAI field)
    """

    # Core OpenAI fields
    id: str = Field(..., description="Job identifier")
    object: Literal["fine_tuning.job"] = Field(default="fine_tuning.job", description="Object type")
    created_at: int = Field(..., description="Unix timestamp of creation")
    finished_at: int | None = Field(
        None, description="Unix timestamp of completion", json_schema_extra=nullable_openai_style
    )
    model: str = Field(..., description="Base model identifier")
    fine_tuned_model: str | None = Field(
        None, description="Fine-tuned model identifier", json_schema_extra=nullable_openai_style
    )
    status: FineTuningJobStatus = Field(..., description="Job status")

    # Training configuration
    training_file: str = Field(..., description="Training dataset identifier")
    validation_file: str | None = Field(
        None, description="Validation dataset identifier", json_schema_extra=nullable_openai_style
    )
    hyperparameters: FineTuningJobHyperparameters | None = Field(
        None,
        description="Training hyperparameters",
        json_schema_extra=lambda s: remove_null_from_anyof(s, add_nullable=False),
    )

    # Results
    trained_tokens: int | None = Field(
        None, description="Total tokens processed", json_schema_extra=nullable_openai_style
    )

    # Error tracking
    error: FineTuningError | None = Field(
        None, description="Error details if failed", json_schema_extra=nullable_openai_style
    )

    # Metadata
    seed: int | None = Field(
        None, description="Random seed", json_schema_extra=lambda s: remove_null_from_anyof(s, add_nullable=False)
    )
    estimated_finish: int | None = Field(
        None, description="Estimated completion timestamp", json_schema_extra=nullable_openai_style
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Custom metadata", json_schema_extra=nullable_openai_style
    )
    organization_id: str | None = Field(
        None,
        description="Organization identifier",
        json_schema_extra=lambda s: remove_null_from_anyof(s, add_nullable=False),
    )

    # Method configuration
    method: FineTuneMethod | None = Field(
        None,
        description="Fine-tuning method configuration",
        json_schema_extra=lambda s: remove_null_from_anyof(s, add_nullable=False),
    )

    # Result files and integrations
    result_files: list[str] = Field(default_factory=list, description="Result file identifiers")
    integrations: list[dict[str, Any]] | None = Field(
        None, description="Integration configs (nullable in OpenAI)", json_schema_extra=nullable_openai_style
    )


@json_schema_type
class CreateFineTuningJobRequest(BaseModel):
    """Request to create a fine-tuning job.

    :param model: Base model to fine-tune
    :param training_file: Training dataset identifier
    :param validation_file: Validation dataset identifier (optional)
    :param hyperparameters: Training hyperparameters (optional)
    :param suffix: Suffix for fine-tuned model name (optional)
    :param method: Fine-tuning method configuration
    :param seed: Random seed for reproducibility (optional)
    :param integrations: Integration configurations (optional)
    :param metadata: Custom metadata (optional)
    """

    model: str = Field(..., description="Base model identifier")
    training_file: str = Field(..., description="Training dataset identifier")
    validation_file: str | None = Field(
        None, description="Validation dataset identifier", json_schema_extra=nullable_openai_style
    )
    hyperparameters: FineTuningJobHyperparameters | None = Field(
        None, description="Training hyperparameters", json_schema_extra=nullable_openai_style
    )
    suffix: str | None = Field(
        None, description="Fine-tuned model name suffix", json_schema_extra=nullable_openai_style
    )
    method: FineTuneMethod = Field(..., description="Fine-tuning method")
    seed: int | None = Field(None, description="Random seed", json_schema_extra=nullable_openai_style)
    integrations: list[dict[str, Any]] | None = Field(
        None, description="Integration configurations", json_schema_extra=nullable_openai_style
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Custom metadata", json_schema_extra=nullable_openai_style
    )


@json_schema_type
class ListFineTuningJobsResponse(BaseModel):
    """Response for listing fine-tuning jobs.

    :param data: List of fine-tuning jobs
    :param has_more: Whether more results are available
    :param object: Object type (always 'list')
    """

    data: list[FineTuningJob] = Field(..., description="List of jobs")
    has_more: bool = Field(default=False, description="More results available")
    object: Literal["list"] = Field(default="list", description="Object type")


@json_schema_type
class ListFineTuningJobEventsResponse(BaseModel):
    """Response for listing fine-tuning job events.

    :param data: List of events
    :param has_more: Whether more results are available
    :param object: Object type (always 'list')
    """

    data: list[FineTuningJobEvent] = Field(..., description="List of events")
    has_more: bool = Field(default=False, description="More results available")
    object: Literal["list"] = Field(default="list", description="Object type")


@json_schema_type
class FineTuningJobCheckpoint(BaseModel):
    """A checkpoint from a fine-tuning job.

    :param id: Checkpoint identifier
    :param created_at: Unix timestamp when checkpoint was created
    :param fine_tuned_model_checkpoint: The name of the fine-tuned checkpoint
    :param step_number: Step number for the checkpoint
    :param metrics: Metrics at this checkpoint
    :param fine_tuning_job_id: The fine-tuning job that created this checkpoint
    :param object: Object type (always 'fine_tuning.job.checkpoint')
    """

    id: str = Field(..., description="Checkpoint identifier")
    created_at: int = Field(..., description="Unix timestamp")
    fine_tuned_model_checkpoint: str = Field(..., description="Checkpoint name")
    step_number: int = Field(..., description="Step number")
    metrics: dict[str, Any] = Field(..., description="Checkpoint metrics")
    fine_tuning_job_id: str = Field(..., description="Fine-tuning job ID")
    object: Literal["fine_tuning.job.checkpoint"] = Field(
        default="fine_tuning.job.checkpoint", description="Object type"
    )


@json_schema_type
class ListFineTuningJobCheckpointsResponse(BaseModel):
    """Response for listing fine-tuning job checkpoints.

    :param data: List of checkpoints
    :param has_more: Whether more results are available
    :param object: Object type (always 'list')
    :param first_id: First checkpoint ID
    :param last_id: Last checkpoint ID
    """

    data: list[FineTuningJobCheckpoint] = Field(..., description="List of checkpoints")
    has_more: bool = Field(default=False, description="More results available")
    object: Literal["list"] = Field(default="list", description="Object type")
    first_id: str | None = Field(None, description="First checkpoint ID", json_schema_extra=nullable_openai_style)
    last_id: str | None = Field(None, description="Last checkpoint ID", json_schema_extra=nullable_openai_style)
