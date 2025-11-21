# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Awaitable, Callable

from ..api import JobStatus, Scheduler
from ..config import InlineSchedulerConfig


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

    async def initialize(self) -> None:
        """Initialize the scheduler and load persisted jobs."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def start(self) -> None:
        """Start processing jobs after all executors are registered."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    def register_job_executor(
        self,
        job_type: str,
        executor: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """Register a job executor function for a specific job type."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def shutdown(self) -> None:
        """Gracefully shutdown the scheduler."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def schedule_job(
        self,
        job_type: str,
        job_data: dict,
        metadata: dict | None = None,
    ) -> str:
        """Schedule a new job for execution."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def get_job_info(self, job_id: str) -> dict:
        """Get complete information about a job."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def delete_job(self, job_id: str) -> bool:
        """Delete a completed or cancelled job."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")

    async def list_jobs(
        self,
        job_type: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List jobs with optional filtering."""
        raise NotImplementedError("Inline scheduler implementation is not yet available")
