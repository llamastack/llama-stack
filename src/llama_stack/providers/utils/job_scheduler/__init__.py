# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .api import JobStatus, Scheduler
from .config import CelerySchedulerConfig, InlineSchedulerConfig, SchedulerConfig


async def scheduler_impl(config: SchedulerConfig) -> Scheduler:
    """
    Factory function to instantiate scheduler implementations.

    Args:
        config: Scheduler configuration (InlineSchedulerConfig or CelerySchedulerConfig)

    Returns:
        Scheduler: An initialized scheduler instance

    Raises:
        ValueError: If the config type is unknown
    """
    impl: Scheduler
    if isinstance(config, InlineSchedulerConfig):
        from .inline import InlineSchedulerImpl

        impl = InlineSchedulerImpl(config)
    elif isinstance(config, CelerySchedulerConfig):
        from .celery import CelerySchedulerImpl

        impl = CelerySchedulerImpl(config)
    else:
        raise ValueError(f"Unknown scheduler config type: {type(config)}")

    await impl.initialize()
    return impl


__all__ = [
    "JobStatus",
    "Scheduler",
    "SchedulerConfig",
    "InlineSchedulerConfig",
    "CelerySchedulerConfig",
    "scheduler_impl",
]
