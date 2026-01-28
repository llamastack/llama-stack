# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    CreateFineTuningJobRequest,
    FineTuningJob,
    ListFineTuningJobCheckpointsResponse,
    ListFineTuningJobEventsResponse,
    ListFineTuningJobsResponse,
)


@runtime_checkable
class FineTuning(Protocol):
    """Fine-tuning API protocol.

    OpenAI-compatible fine-tuning interface for training models.
    """

    async def create_fine_tuning_job(self, request: CreateFineTuningJobRequest) -> FineTuningJob:
        """Create a new fine-tuning job."""
        ...

    async def list_fine_tuning_jobs(self, after: str | None = None, limit: int = 20) -> ListFineTuningJobsResponse:
        """List fine-tuning jobs with pagination."""
        ...

    async def retrieve_fine_tuning_job(self, fine_tuning_job_id: str) -> FineTuningJob:
        """Retrieve a specific fine-tuning job by ID."""
        ...

    async def cancel_fine_tuning_job(self, fine_tuning_job_id: str) -> FineTuningJob:
        """Cancel a fine-tuning job."""
        ...

    async def list_fine_tuning_job_checkpoints(
        self, fine_tuning_job_id: str, after: str | None = None, limit: int = 20
    ) -> ListFineTuningJobCheckpointsResponse:
        """List checkpoints for a fine-tuning job."""
        ...

    async def list_fine_tuning_events(
        self, fine_tuning_job_id: str, after: str | None = None, limit: int = 20
    ) -> ListFineTuningJobEventsResponse:
        """List events/logs for a fine-tuning job."""
        ...
