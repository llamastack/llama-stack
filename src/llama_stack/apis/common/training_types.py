# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PostTrainingMetric(BaseModel):
    """Training metrics captured during post-training jobs.

    :param epoch: Training epoch number
    :param train_loss: Loss value on the training dataset
    :param validation_loss: Loss value on the validation dataset
    :param perplexity: Perplexity metric indicating model confidence
    """

    epoch: int
    train_loss: float
    validation_loss: float
    perplexity: float


@json_schema_type
class Checkpoint(BaseModel):
    """Checkpoint created during training runs.

    :param identifier: Unique identifier for the checkpoint
    :param created_at: Timestamp when the checkpoint was created
    :param epoch: Training epoch when the checkpoint was saved
    :param post_training_job_id: Identifier of the training job that created this checkpoint
    :param path: File system path where the checkpoint is stored
    :param training_metrics: (Optional) Training metrics associated with this checkpoint
    """

    identifier: str
    created_at: datetime
    epoch: int
    post_training_job_id: str
    path: str
    training_metrics: PostTrainingMetric | None = None
