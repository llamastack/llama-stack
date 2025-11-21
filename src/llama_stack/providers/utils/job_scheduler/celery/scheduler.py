# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Awaitable, Callable

from ..api import JobStatus, Scheduler
from ..config import CelerySchedulerConfig


class CelerySchedulerImpl(Scheduler):
    """
    Celery-based scheduler implementation for distributed multi-worker deployments.

    This scheduler uses Celery to distribute jobs across multiple worker processes
    or machines. It provides:
    - Persistent job queue (via Redis/RabbitMQ broker)
    - Multi-worker coordination
    - Crash recovery (jobs survive server restarts)
    - Distributed task execution

    This is suitable for:
    - Production deployments
    - Multi-worker scenarios
    - High availability requirements
    """

    def __init__(self, config: CelerySchedulerConfig):
        self.config = config
        self._job_executors: dict[str, Callable[[dict], Awaitable[dict]]] = {}

    def register_job_executor(
        self,
        job_type: str,
        executor: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """Register a job executor function for a specific job type."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def initialize(self) -> None:
        """Initialize the Celery scheduler."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def start(self) -> None:
        """Start processing jobs after all executors are registered."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def shutdown(self) -> None:
        """Gracefully shutdown the Celery scheduler."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def schedule_job(
        self,
        job_type: str,
        job_data: dict,
        metadata: dict | None = None,
    ) -> str:
        """Schedule a new job for execution via Celery."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def get_job_info(self, job_id: str) -> dict:
        """Get complete information about a job from Celery result backend."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running Celery job."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def delete_job(self, job_id: str) -> bool:
        """Delete a completed or cancelled job from Celery result backend."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List jobs from Celery result backend with optional filtering."""
        raise NotImplementedError("Celery scheduler implementation is not yet available")
