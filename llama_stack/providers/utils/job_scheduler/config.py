# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference


class SchedulerConfig(BaseModel):
    """Base class for scheduler configurations."""

    type: str


class InlineSchedulerConfig(SchedulerConfig):
    """
    Configuration for inline (asyncio-based) scheduler.

    This scheduler runs jobs in the same process using asyncio tasks.
    Suitable for development and single-worker deployments.
    """

    type: Literal["inline"] = "inline"
    kvstore: KVStoreReference = Field(
        description="KVStore reference for persisting job state",
    )
    max_concurrent_jobs: int = Field(
        default=10,
        description="Maximum number of jobs that can run concurrently",
    )


class CelerySchedulerConfig(SchedulerConfig):
    """
    Configuration for Celery-based distributed scheduler.

    This scheduler distributes jobs across multiple worker processes/machines.
    Suitable for production and multi-worker deployments.
    """

    type: Literal["celery"] = "celery"
    broker_url: str = Field(
        description="Celery broker URL (e.g., 'redis://localhost:6379/0')",
    )
    result_backend: str = Field(
        description="Celery result backend URL (e.g., 'redis://localhost:6379/1')",
    )


SchedulerConfigUnion = Annotated[
    InlineSchedulerConfig | CelerySchedulerConfig,
    Field(discriminator="type"),
]
