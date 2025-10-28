# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import warnings
from datetime import datetime
from typing import Any, Literal

import aiohttp
from pydantic import BaseModel, ConfigDict

from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.providers.remote.post_training.nvidia.config import NvidiaPostTrainingConfig
from llama_stack.providers.remote.post_training.nvidia.utils import warn_unsupported_params
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

from .models import _MODEL_ENTRIES

# Map API status to JobStatus enum
STATUS_MAPPING = {
    "running": JobStatus.in_progress.value,
    "completed": JobStatus.completed.value,
    "failed": JobStatus.failed.value,
    "cancelled": JobStatus.cancelled.value,
    "pending": JobStatus.scheduled.value,
    "unknown": JobStatus.scheduled.value,
}


class NvidiaPostTrainingJob(PostTrainingJob):
    """Parse the response from the Customizer API.
    Inherits job_uuid from PostTrainingJob.
    Adds status, created_at, updated_at parameters.
    Passes through all other parameters from data field in the response.
    """

    model_config = ConfigDict(extra="allow")
    status: JobStatus
    created_at: datetime
    updated_at: datetime


class ListNvidiaPostTrainingJobs(BaseModel):
    data: list[NvidiaPostTrainingJob]


class NvidiaPostTrainingJobStatusResponse(PostTrainingJobStatusResponse):
    model_config = ConfigDict(extra="allow")


class NvidiaPostTrainingAdapter(ModelRegistryHelper):
    def __init__(self, config: NvidiaPostTrainingConfig):
        self.config = config
        self.headers = {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        self.timeout = aiohttp.ClientTimeout(total=config.timeout)
        # TODO: filter by available models based on /config endpoint
        ModelRegistryHelper.__init__(self, model_entries=_MODEL_ENTRIES)
        self.session = None

        self.customizer_url = config.customizer_url
        if not self.customizer_url:
            warnings.warn("Customizer URL is not set, using default value: http://nemo.test", stacklevel=2)
            self.customizer_url = "http://nemo.test"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
        return self.session

    async def _make_request(
        self,
        method: str,
        path: str,
        headers: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Helper method to make HTTP requests to the Customizer API."""
        url = f"{self.customizer_url}{path}"
        request_headers = self.headers.copy()

        if headers:
            request_headers.update(headers)

        # Add content-type header for JSON requests
        if json and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        session = await self._get_session()
        for _ in range(self.config.max_retries):
            async with session.request(method, url, params=params, json=json, **kwargs) as response:
                if response.status >= 400:
                    error_data = await response.json()
                    raise Exception(f"API request failed: {error_data}")
                return await response.json()

    async def get_training_jobs(
        self,
        page: int | None = 1,
        page_size: int | None = 10,
        sort: Literal["created_at", "-created_at"] | None = "created_at",
    ) -> ListNvidiaPostTrainingJobs:
        """Get all customization jobs.
        Updated the base class return type from ListPostTrainingJobsResponse to ListNvidiaPostTrainingJobs.

        Returns a ListNvidiaPostTrainingJobs object with the following fields:
            - data: List[NvidiaPostTrainingJob] - List of NvidiaPostTrainingJob objects

        ToDo: Support for schema input for filtering.
        """
        params = {"page": page, "page_size": page_size, "sort": sort}

        response = await self._make_request("GET", "/v1/customization/jobs", params=params)

        jobs = []
        for job in response.get("data", []):
            job_id = job.pop("id")
            job_status = job.pop("status", "scheduled").lower()
            mapped_status = STATUS_MAPPING.get(job_status, "scheduled")

            # Convert string timestamps to datetime objects
            created_at = (
                datetime.fromisoformat(job.pop("created_at"))
                if "created_at" in job
                else datetime.now(tz=datetime.timezone.utc)
            )
            updated_at = (
                datetime.fromisoformat(job.pop("updated_at"))
                if "updated_at" in job
                else datetime.now(tz=datetime.timezone.utc)
            )

            # Create NvidiaPostTrainingJob instance
            jobs.append(
                NvidiaPostTrainingJob(
                    job_uuid=job_id,
                    status=JobStatus(mapped_status),
                    created_at=created_at,
                    updated_at=updated_at,
                    **job,
                )
            )

        return ListNvidiaPostTrainingJobs(data=jobs)

    async def get_training_job_status(self, job_uuid: str) -> NvidiaPostTrainingJobStatusResponse:
        """Get the status of a customization job.
        Updated the base class return type from PostTrainingJobResponse to NvidiaPostTrainingJob.

        Returns a NvidiaPostTrainingJob object with the following fields:
            - job_uuid: str - Unique identifier for the job
            - status: JobStatus - Current status of the job (in_progress, completed, failed, cancelled, scheduled)
            - created_at: datetime - The time when the job was created
            - updated_at: datetime - The last time the job status was updated

        Additional fields that may be included:
            - steps_completed: Optional[int] - Number of training steps completed
            - epochs_completed: Optional[int] - Number of epochs completed
            - percentage_done: Optional[float] - Percentage of training completed (0-100)
            - best_epoch: Optional[int] - The epoch with the best performance
            - train_loss: Optional[float] - Training loss of the best checkpoint
            - val_loss: Optional[float] - Validation loss of the best checkpoint
            - metrics: Optional[Dict] - Additional training metrics
            - status_logs: Optional[List] - Detailed logs of status changes
        """
        response = await self._make_request(
            "GET",
            f"/v1/customization/jobs/{job_uuid}/status",
            params={"job_id": job_uuid},
        )

        api_status = response.pop("status").lower()
        mapped_status = STATUS_MAPPING.get(api_status, "scheduled")

        return NvidiaPostTrainingJobStatusResponse(
            status=JobStatus(mapped_status),
            job_uuid=job_uuid,
            started_at=datetime.fromisoformat(response.pop("created_at")),
            updated_at=datetime.fromisoformat(response.pop("updated_at")),
            **response,
        )

    async def cancel_training_job(self, job_uuid: str) -> None:
        await self._make_request(
            method="POST", path=f"/v1/customization/jobs/{job_uuid}/cancel", params={"job_id": job_uuid}
        )

    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse:
        raise NotImplementedError("Job artifacts are not implemented yet")

    async def get_post_training_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse:
        raise NotImplementedError("Job artifacts are not implemented yet")

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: dict[str, Any],
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> NvidiaPostTrainingJob:
        """
        Fine-tunes a model on a dataset.
        Currently only supports Lora finetuning for standlone docker container.
        Assumptions:
            - nemo microservice is running and endpoint is set in config.customizer_url
            - dataset is registered separately in nemo datastore
            - model checkpoint is downloaded as per nemo customizer requirements

        Parameters:
            training_config: TrainingConfig - Configuration for training
            model: str - NeMo Customizer configuration name
            algorithm_config: Optional[AlgorithmConfig] - Algorithm-specific configuration
            checkpoint_dir: Optional[str] - Directory containing model checkpoints, ignored atm
            job_uuid: str - Unique identifier for the job, ignored atm
            hyperparam_search_config: Dict[str, Any] - Configuration for hyperparameter search, ignored atm
            logger_config: Dict[str, Any] - Configuration for logging, ignored atm

        Environment Variables:
            - NVIDIA_API_KEY: str - API key for the NVIDIA API
                Default: None
            - NVIDIA_DATASET_NAMESPACE: str - Namespace of the dataset
                Default: "default"
            - NVIDIA_CUSTOMIZER_URL: str - URL of the NeMo Customizer API
                Default: "http://nemo.test"
            - NVIDIA_PROJECT_ID: str - ID of the project
                Default: "test-project"
            - NVIDIA_OUTPUT_MODEL_DIR: str - Directory to save the output model
                Default: "test-example-model@v1"

        Supported models:
            - meta/llama-3.1-8b-instruct
            - meta/llama-3.2-1b-instruct

        Supported algorithm configs:
            - LoRA, SFT

        Supported Parameters:
            - TrainingConfig:
                - n_epochs: int - Number of epochs to train
                    Default: 50
                - data_config: DataConfig - Configuration for the dataset
                - optimizer_config: OptimizerConfig - Configuration for the optimizer
                - dtype: str - Data type for training
                    not supported (users are informed via warnings)
                - efficiency_config: EfficiencyConfig - Configuration for efficiency
                    not supported
                - max_steps_per_epoch: int - Maximum number of steps per epoch
                    Default: 1000
                ## NeMo customizer specific parameters
                - log_every_n_steps: int - Log every n steps
                    Default: None
                - val_check_interval: float - Validation check interval
                    Default: 0.25
                - sequence_packing_enabled: bool - Sequence packing enabled
                    Default: False
                ## NeMo customizer specific SFT parameters
                - hidden_dropout: float - Hidden dropout
                    Default: None (0.0-1.0)
                - attention_dropout: float - Attention dropout
                    Default: None (0.0-1.0)
                - ffn_dropout: float - FFN dropout
                    Default: None (0.0-1.0)

            - DataConfig:
                - dataset_id: str - Dataset ID
                - batch_size: int - Batch size
                    Default: 8

            - OptimizerConfig:
                - lr: float - Learning rate
                    Default: 0.0001
                ## NeMo customizer specific parameter
                - weight_decay: float - Weight decay
                    Default: 0.01

            - LoRA config:
                ## NeMo customizer specific LoRA parameters
                - alpha: int - Scaling factor for the LoRA update
                    Default: 16
            Note:
                - checkpoint_dir, hyperparam_search_config, logger_config are not supported (users are informed via warnings)
                - Some parameters from TrainingConfig, DataConfig, OptimizerConfig are not supported (users are informed via warnings)

            User is informed about unsupported parameters via warnings.
        """

        # Check for unsupported method parameters
        unsupported_method_params = []
        if checkpoint_dir:
            unsupported_method_params.append(f"checkpoint_dir={checkpoint_dir}")
        if hyperparam_search_config:
            unsupported_method_params.append("hyperparam_search_config")
        if logger_config:
            unsupported_method_params.append("logger_config")

        if unsupported_method_params:
            warnings.warn(
                f"Parameters: {', '.join(unsupported_method_params)} are not supported and will be ignored",
                stacklevel=2,
            )

        # Define all supported parameters
        supported_params = {
            "training_config": {
                "n_epochs",
                "data_config",
                "optimizer_config",
                "log_every_n_steps",
                "val_check_interval",
                "sequence_packing_enabled",
                "hidden_dropout",
                "attention_dropout",
                "ffn_dropout",
            },
            "data_config": {"dataset_id", "batch_size"},
            "optimizer_config": {"lr", "weight_decay"},
            "lora_config": {"type", "alpha"},
        }

        # Validate all parameters at once
        warn_unsupported_params(training_config, supported_params["training_config"], "TrainingConfig")
        warn_unsupported_params(training_config["data_config"], supported_params["data_config"], "DataConfig")
        warn_unsupported_params(
            training_config["optimizer_config"], supported_params["optimizer_config"], "OptimizerConfig"
        )

        output_model = self.config.output_model_dir

        # Prepare base job configuration
        job_config = {
            "config": model,
            "dataset": {
                "name": training_config["data_config"]["dataset_id"],
                "namespace": self.config.dataset_namespace,
            },
            "hyperparameters": {
                "training_type": "sft",
                "finetuning_type": "lora",
                **{
                    k: v
                    for k, v in {
                        "epochs": training_config.get("n_epochs"),
                        "batch_size": training_config["data_config"].get("batch_size"),
                        "learning_rate": training_config["optimizer_config"].get("lr"),
                        "weight_decay": training_config["optimizer_config"].get("weight_decay"),
                        "log_every_n_steps": training_config.get("log_every_n_steps"),
                        "val_check_interval": training_config.get("val_check_interval"),
                        "sequence_packing_enabled": training_config.get("sequence_packing_enabled"),
                    }.items()
                    if v is not None
                },
            },
            "project": self.config.project_id,
            # TODO: ignored ownership, add it later
            # "ownership": {"created_by": self.config.user_id, "access_policies": self.config.access_policies},
            "output_model": output_model,
        }

        # Handle SFT-specific optional parameters
        job_config["hyperparameters"]["sft"] = {
            k: v
            for k, v in {
                "ffn_dropout": training_config.get("ffn_dropout"),
                "hidden_dropout": training_config.get("hidden_dropout"),
                "attention_dropout": training_config.get("attention_dropout"),
            }.items()
            if v is not None
        }

        # Remove the sft dictionary if it's empty
        if not job_config["hyperparameters"]["sft"]:
            job_config["hyperparameters"].pop("sft")

        # Handle LoRA-specific configuration
        if algorithm_config:
            if algorithm_config.type == "LoRA":
                warn_unsupported_params(algorithm_config, supported_params["lora_config"], "LoRA config")
                job_config["hyperparameters"]["lora"] = {
                    k: v for k, v in {"alpha": algorithm_config.alpha}.items() if v is not None
                }
            else:
                raise NotImplementedError(f"Unsupported algorithm config: {algorithm_config}")

        # Create the customization job
        response = await self._make_request(
            method="POST",
            path="/v1/customization/jobs",
            headers={"Accept": "application/json"},
            json=job_config,
        )

        job_uuid = response["id"]
        response.pop("status")
        created_at = datetime.fromisoformat(response.pop("created_at"))
        updated_at = datetime.fromisoformat(response.pop("updated_at"))

        return NvidiaPostTrainingJob(
            job_uuid=job_uuid, status=JobStatus.in_progress, created_at=created_at, updated_at=updated_at, **response
        )

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        """Optimize a model based on preference data."""
        raise NotImplementedError("Preference optimization is not implemented yet")

    async def get_training_job_container_logs(self, job_uuid: str) -> PostTrainingJobStatusResponse:
        raise NotImplementedError("Job logs are not implemented yet")
