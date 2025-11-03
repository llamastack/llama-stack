# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PostTrainingMetric(BaseModel):
    """Training metrics captured during post-training jobs."""

    epoch: int = Field(description="Training epoch number.")
    train_loss: float = Field(description="Loss value on the training dataset.")
    validation_loss: float = Field(description="Loss value on the validation dataset.")
    perplexity: float = Field(description="Perplexity metric indicating model confidence.")


@json_schema_type
class Checkpoint(BaseModel):
    """Checkpoint created during training runs."""

    identifier: str = Field(description="Unique identifier for the checkpoint.")
    created_at: datetime = Field(description="Timestamp when the checkpoint was created.")
    epoch: int = Field(description="Training epoch when the checkpoint was saved.")
    post_training_job_id: str = Field(description="Identifier of the training job that created this checkpoint.")
    path: str = Field(description="File system path where the checkpoint is stored.")
    training_metrics: PostTrainingMetric | None = Field(description="Training metrics associated with this checkpoint.")
