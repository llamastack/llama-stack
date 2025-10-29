# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Scheduler(Protocol):
    """
    Abstract scheduler protocol for managing async jobs.

    This provides a pluggable backend for job scheduling and execution.
    """

    async def schedule_job(
        self,
        job_type: str,
        job_data: dict,
        metadata: dict | None = None,
    ) -> str:
        """
        Schedule a new job for execution.

        Args:
            job_type: Type of job (e.g., "batch_processing", "log_aggregation", "metrics_collection")
            job_data: Job-specific data and parameters
            metadata: Additional metadata for tracking

        Returns:
            job_id: UUID for tracking the job
        """
        ...

    async def get_job_info(self, job_id: str) -> dict:
        """
        Get complete information about a job.

        Returns:
            {
                "job_id": str,
                "job_type": str,
                "status": JobStatus,
                "created_at": datetime,
                "started_at": datetime | None,
                "completed_at": datetime | None,
                "progress": float,  # 0.0 to 1.0
                "metadata": dict,
                "error": str | None,  # Error message if status == FAILED
                "result": dict | None,  # Job result if status == COMPLETED
            }
        """
        ...

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.

        Args:
            job_id: Job to cancel

        Returns:
            True if job was cancelled, False if not found or already completed
        """
        ...

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a completed or cancelled job.

        Args:
            job_id: Job to delete

        Returns:
            True if job was deleted, False if not found
        """
        ...

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        List jobs with optional filtering.

        Args:
            job_type: Filter by job type (e.g., "batch_processing", "log_aggregation", "metrics_collection")
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Offset for pagination

        Returns:
            List of job info dictionaries
        """
        ...

    def register_job_executor(
        self,
        job_type: str,
        executor: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """
        Register a job executor function for a specific job type.

        This allows components to register custom job execution logic with the scheduler.
        When a job of the registered type is executed, the scheduler will call the
        registered executor function.

        Args:
            job_type: The type of job (e.g., "vector_store_file_batch")
            executor: Async function that takes job_data dict and returns result dict
        """
        ...

    async def initialize(self) -> None:
        """Initialize the scheduler (connect to backend, etc.)"""
        ...

    async def shutdown(self) -> None:
        """Gracefully shutdown the scheduler"""
        ...
