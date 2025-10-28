# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from typing import Any

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    Checkpoint,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.providers.inline.post_training.huggingface.config import (
    HuggingFacePostTrainingConfig,
)
from llama_stack.providers.utils.scheduler import JobArtifact, Scheduler
from llama_stack.providers.utils.scheduler import JobStatus as SchedulerJobStatus


class TrainingArtifactType(Enum):
    CHECKPOINT = "checkpoint"
    RESOURCES_STATS = "resources_stats"


_JOB_TYPE_SUPERVISED_FINE_TUNE = "supervised-fine-tune"
_JOB_TYPE_DPO_TRAINING = "dpo-training"


class HuggingFacePostTrainingImpl:
    def __init__(
        self,
        config: HuggingFacePostTrainingConfig,
        datasetio_api: DatasetIO,
        datasets: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets
        self._scheduler = Scheduler()

    async def shutdown(self) -> None:
        await self._scheduler.shutdown()

    @staticmethod
    def _checkpoint_to_artifact(checkpoint: Checkpoint) -> JobArtifact:
        return JobArtifact(
            type=TrainingArtifactType.CHECKPOINT.value,
            name=checkpoint.identifier,
            uri=checkpoint.path,
            metadata=dict(checkpoint),
        )

    @staticmethod
    def _resources_stats_to_artifact(resources_stats: dict[str, Any]) -> JobArtifact:
        return JobArtifact(
            type=TrainingArtifactType.RESOURCES_STATS.value,
            name=TrainingArtifactType.RESOURCES_STATS.value,
            metadata=resources_stats,
        )

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        async def handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb):
            from llama_stack.providers.inline.post_training.huggingface.recipes.finetune_single_device import (
                HFFinetuningSingleDevice,
            )

            on_log_message_cb("Starting HF finetuning")

            recipe = HFFinetuningSingleDevice(
                job_uuid=job_uuid,
                datasetio_api=self.datasetio_api,
                datasets_api=self.datasets_api,
            )

            resources_allocated, checkpoints = await recipe.train(
                model=model,
                output_dir=checkpoint_dir,
                job_uuid=job_uuid,
                lora_config=algorithm_config,
                config=training_config,
                provider_config=self.config,
            )

            on_artifact_collected_cb(self._resources_stats_to_artifact(resources_allocated))
            if checkpoints:
                for checkpoint in checkpoints:
                    artifact = self._checkpoint_to_artifact(checkpoint)
                    on_artifact_collected_cb(artifact)

            on_status_change_cb(SchedulerJobStatus.completed)
            on_log_message_cb("HF finetuning completed")

        job_uuid = self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, handler)
        return PostTrainingJob(job_uuid=job_uuid)

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        async def handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb):
            from llama_stack.providers.inline.post_training.huggingface.recipes.finetune_single_device_dpo import (
                HFDPOAlignmentSingleDevice,
            )

            on_log_message_cb("Starting HF DPO alignment")

            recipe = HFDPOAlignmentSingleDevice(
                job_uuid=job_uuid,
                datasetio_api=self.datasetio_api,
                datasets_api=self.datasets_api,
            )

            resources_allocated, checkpoints = await recipe.train(
                model=finetuned_model,
                output_dir=f"{self.config.dpo_output_dir}/{job_uuid}",
                job_uuid=job_uuid,
                dpo_config=algorithm_config,
                config=training_config,
                provider_config=self.config,
            )

            on_artifact_collected_cb(self._resources_stats_to_artifact(resources_allocated))
            if checkpoints:
                for checkpoint in checkpoints:
                    artifact = self._checkpoint_to_artifact(checkpoint)
                    on_artifact_collected_cb(artifact)
            else:
                on_log_message_cb("Warning: No checkpoints were saved during DPO training")

            on_status_change_cb(SchedulerJobStatus.completed)
            on_log_message_cb("HF DPO alignment completed")

        job_uuid = self._scheduler.schedule(_JOB_TYPE_DPO_TRAINING, job_uuid, handler)
        return PostTrainingJob(job_uuid=job_uuid)

    @staticmethod
    def _get_artifacts_metadata_by_type(job, artifact_type):
        return [artifact.metadata for artifact in job.artifacts if artifact.type == artifact_type]

    @classmethod
    def _get_checkpoints(cls, job):
        return cls._get_artifacts_metadata_by_type(job, TrainingArtifactType.CHECKPOINT.value)

    @classmethod
    def _get_resources_allocated(cls, job):
        data = cls._get_artifacts_metadata_by_type(job, TrainingArtifactType.RESOURCES_STATS.value)
        return data[0] if data else None

    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        job = self._scheduler.get_job(job_uuid)

        match job.status:
            # TODO: Add support for other statuses to API
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                raise NotImplementedError()

        return PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=status,
            scheduled_at=job.scheduled_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            checkpoints=self._get_checkpoints(job),
            resources_allocated=self._get_resources_allocated(job),
        )

    async def cancel_training_job(self, job_uuid: str) -> None:
        self._scheduler.cancel(job_uuid)

    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        job = self._scheduler.get_job(job_uuid)
        return PostTrainingJobArtifactsResponse(job_uuid=job_uuid, checkpoints=self._get_checkpoints(job))

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(
            data=[PostTrainingJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )
