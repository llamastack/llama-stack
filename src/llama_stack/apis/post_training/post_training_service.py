# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)


@runtime_checkable
@trace_protocol
class PostTrainingService(Protocol):
    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str | None = None,
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        """Run supervised fine-tuning of a model."""
        ...

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        """Run preference optimization of a model."""
        ...

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """Get all training jobs."""
        ...

    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse:
        """Get the status of a training job."""
        ...

    async def cancel_training_job(self, job_uuid: str) -> None:
        """Cancel a training job."""
        ...

    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse:
        """Get the artifacts of a training job."""
        ...
