# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    CancelTrainingJobRequest,
    GetTrainingJobArtifactsRequest,
    GetTrainingJobStatusRequest,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    PreferenceOptimizeRequest,
    SupervisedFineTuneRequest,
)


@runtime_checkable
class PostTraining(Protocol):
    async def supervised_fine_tune(self, request: SupervisedFineTuneRequest) -> PostTrainingJob: ...

    async def preference_optimize(self, request: PreferenceOptimizeRequest) -> PostTrainingJob: ...

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse: ...

    async def get_training_job_status(self, request: GetTrainingJobStatusRequest) -> PostTrainingJobStatusResponse: ...

    async def cancel_training_job(self, request: CancelTrainingJobRequest) -> None: ...

    async def get_training_job_artifacts(
        self, request: GetTrainingJobArtifactsRequest
    ) -> PostTrainingJobArtifactsResponse: ...
