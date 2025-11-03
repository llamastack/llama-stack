# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.common.training_types import Checkpoint

from . import routes  # noqa: F401
from .models import (
    AlgorithmConfig,
    DataConfig,
    DatasetFormat,
    DPOAlignmentConfig,
    DPOLossType,
    EfficiencyConfig,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    OptimizerConfig,
    OptimizerType,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobLogStream,
    PostTrainingJobStatusResponse,
    PostTrainingRLHFRequest,
    PreferenceOptimizeRequest,
    QATFinetuningConfig,
    RLHFAlgorithm,
    SupervisedFineTuneRequest,
    TrainingConfig,
)
from .post_training_service import PostTrainingService

# Backward compatibility - export PostTraining as alias for PostTrainingService
PostTraining = PostTrainingService

__all__ = [
    "PostTraining",
    "PostTrainingService",
    "Checkpoint",
    "JobStatus",
    "OptimizerType",
    "DatasetFormat",
    "DataConfig",
    "OptimizerConfig",
    "EfficiencyConfig",
    "TrainingConfig",
    "LoraFinetuningConfig",
    "QATFinetuningConfig",
    "AlgorithmConfig",
    "PostTrainingJobLogStream",
    "RLHFAlgorithm",
    "DPOLossType",
    "DPOAlignmentConfig",
    "PostTrainingRLHFRequest",
    "PostTrainingJob",
    "PostTrainingJobStatusResponse",
    "ListPostTrainingJobsResponse",
    "PostTrainingJobArtifactsResponse",
    "SupervisedFineTuneRequest",
    "PreferenceOptimizeRequest",
]
