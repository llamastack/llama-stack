# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from llama_stack.core.utils.serialize import EnumEncoder
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore

from ..api import JobStatus, Scheduler
from ..config import InlineSchedulerConfig

JOB_PREFIX = "job_scheduler:jobs:"


class InlineSchedulerImpl(Scheduler):
    """
    Inline scheduler implementation using asyncio for single-worker deployments.

    This scheduler runs jobs in the same process using asyncio tasks. Jobs and their
    state are persisted to KVStore for crash recovery.

    This is suitable for:
    - Development and testing
    - Single-worker deployments
    - Scenarios where external dependencies (Redis, RabbitMQ) are not available
    """

    def __init__(self, config: InlineSchedulerConfig):
        self.config = config
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore: asyncio.Semaphore
        self._shutdown_event = asyncio.Event()
        self._kvstore: KVStore
        self._job_executors: dict[str, Callable[[dict], Awaitable[dict]]] = {}

    async def initialize(self) -> None:
        """Initialize the scheduler and load persisted jobs."""
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)

        # Initialize KVStore
        self._kvstore = await kvstore_impl(self.config.kvstore)

        # Load persisted jobs from KVStore
        await self._load_jobs_from_storage()

        # Resume incomplete jobs
        await self._resume_incomplete_jobs()

    def register_job_executor(
        self,
        job_type: str,
        executor: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """Register a job executor function for a specific job type."""
        self._job_executors[job_type] = executor

    async def _load_jobs_from_storage(self) -> None:
        """Load all jobs from KVStore into memory."""
        start_key = JOB_PREFIX
        end_key = f"{JOB_PREFIX}\xff"

        stored_values = await self._kvstore.values_in_range(start_key, end_key)

        for value in stored_values:
            job = json.loads(value)
            # Deserialize datetime strings back to datetime objects
            job["created_at"] = datetime.fromisoformat(job["created_at"])
            if job.get("started_at"):
                job["started_at"] = datetime.fromisoformat(job["started_at"])
            if job.get("completed_at"):
                job["completed_at"] = datetime.fromisoformat(job["completed_at"])
            job["status"] = JobStatus(job["status"])

            self._jobs[job["job_id"]] = job

    async def _resume_incomplete_jobs(self) -> None:
        """Resume jobs that were running when server crashed."""
        for job_id, job in self._jobs.items():
            if job["status"] in [JobStatus.PENDING, JobStatus.RUNNING]:
                # Reset running jobs to pending
                if job["status"] == JobStatus.RUNNING:
                    job["status"] = JobStatus.PENDING
                    job["started_at"] = None
                    await self._save_job_to_storage(job)

                # Restart the job
                task = asyncio.create_task(self._run_job(job_id))
                self._tasks[job_id] = task

    async def _save_job_to_storage(self, job: dict[str, Any]) -> None:
        """Persist job to KVStore."""
        key = f"{JOB_PREFIX}{job['job_id']}"
        await self._kvstore.set(key, json.dumps(job, cls=EnumEncoder))

    async def _delete_job_from_storage(self, job_id: str) -> None:
        """Delete job from KVStore."""
        key = f"{JOB_PREFIX}{job_id}"
        await self._kvstore.delete(key)

    async def shutdown(self) -> None:
        """Gracefully shutdown the scheduler."""
        self._shutdown_event.set()

        # Cancel all running tasks
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self._tasks.clear()

    async def schedule_job(
        self,
        job_type: str,
        job_data: dict,
        metadata: dict | None = None,
    ) -> str:
        """Schedule a new job for execution."""
        job_id = str(uuid.uuid4())

        job_info = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.PENDING,
            "created_at": datetime.now(UTC),
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "metadata": metadata or {},
            "job_data": job_data,
            "error": None,
            "result": None,
        }

        self._jobs[job_id] = job_info

        # Persist to KVStore
        await self._save_job_to_storage(job_info)

        # Create and schedule the task
        task = asyncio.create_task(self._run_job(job_id))
        self._tasks[job_id] = task

        return job_id

    async def _run_job(self, job_id: str) -> None:
        """Run a job asynchronously."""
        job = self._jobs[job_id]

        try:
            # Acquire semaphore to limit concurrent jobs
            async with self._semaphore:
                # Update status to RUNNING
                job["status"] = JobStatus.RUNNING
                job["started_at"] = datetime.now(UTC)
                await self._save_job_to_storage(job)

                # Execute the job based on job_type
                result = await self._execute_job(job)

                # Mark as completed
                job["status"] = JobStatus.COMPLETED
                job["completed_at"] = datetime.now(UTC)
                job["progress"] = 1.0
                job["result"] = result
                await self._save_job_to_storage(job)

        except asyncio.CancelledError:
            # Job was cancelled
            job["status"] = JobStatus.CANCELLED
            job["completed_at"] = datetime.now(UTC)
            await self._save_job_to_storage(job)
            raise

        except Exception as e:
            # Job failed
            job["status"] = JobStatus.FAILED
            job["completed_at"] = datetime.now(UTC)
            job["error"] = str(e)
            job["result"] = {"error_details": traceback.format_exc()}
            await self._save_job_to_storage(job)

        finally:
            # Clean up task reference
            if job_id in self._tasks:
                del self._tasks[job_id]

    async def _execute_job(self, job: dict) -> dict:
        """
        Execute a job based on its type.

        If a custom executor is registered for the job type, it will be called.
        Otherwise, raises an error for unknown job types.
        """
        job_type = job["job_type"]
        job_data = job["job_data"]

        # Check if a custom executor is registered for this job type
        if job_type in self._job_executors:
            executor = self._job_executors[job_type]
            return await executor(job_data)

        raise ValueError(f"No executor registered for job type: {job_type}")

    async def get_job_info(self, job_id: str) -> dict:
        """Get complete information about a job."""
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs[job_id].copy()

        # Remove internal job_data field from response
        job.pop("job_data", None)

        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]

        # Can only cancel pending or running jobs
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        # Cancel the task if it exists
        if job_id in self._tasks:
            task = self._tasks[job_id]
            if not task.done():
                task.cancel()

        # Update job status
        job["status"] = JobStatus.CANCELLED
        job["completed_at"] = datetime.now(UTC)
        await self._save_job_to_storage(job)

        return True

    async def delete_job(self, job_id: str) -> bool:
        """Delete a completed or cancelled job."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]

        # Can only delete completed, failed, or cancelled jobs
        if job["status"] in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False

        # Remove from memory and storage
        del self._jobs[job_id]
        await self._delete_job_from_storage(job_id)

        return True

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())

        # Filter by job_type
        if job_type is not None:
            jobs = [j for j in jobs if j["job_type"] == job_type]

        # Filter by status
        if status is not None:
            jobs = [j for j in jobs if j["status"] == status]

        # Sort by created_at (newest first)
        jobs.sort(key=lambda j: j["created_at"], reverse=True)

        # Apply pagination
        jobs = jobs[offset : offset + limit]

        # Convert to return format (remove internal fields)
        result = []
        for job in jobs:
            job_copy = job.copy()
            # Remove internal job_data field
            job_copy.pop("job_data", None)
            result.append(job_copy)

        return result
