# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ogx.log import get_logger

logger = get_logger(name=__name__, category="providers::utils")

T = TypeVar("T")


class QueueFullError(Exception):
    """Raised when the scheduler's wait queue has reached its maximum size."""

    pass


class AsyncRequestScheduler:
    """Limits concurrent async operations with FIFO queuing.

    Prevents overwhelming backends (like Docling Serve) that degrade or
    crash under unbounded concurrency. Callers await ``submit()`` which
    blocks until a concurrency slot is available, then executes the work.
    """

    def __init__(
        self,
        max_concurrency: int = 2,
        max_queue_size: int = 0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency
        self._max_queue_size = max_queue_size
        self._active = 0
        self._queued = 0

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return self._queued

    async def submit(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute *fn* when a concurrency slot is available.

        If all slots are busy the call queues in FIFO order until one
        frees up. When *max_queue_size* is non-zero and the queue is
        full, raises ``QueueFullError`` immediately.

        *fn* must be a zero-argument async callable (e.g. a lambda or
        partial wrapping the real call). This avoids eagerly creating
        coroutine objects that would start executing before we acquire
        the semaphore.
        """
        if self._max_queue_size and self._queued >= self._max_queue_size:
            raise QueueFullError(f"Failed to schedule request: queue is full ({self._queued}/{self._max_queue_size})")

        self._queued += 1
        try:
            await self._semaphore.acquire()
        except BaseException:
            self._queued -= 1
            raise

        self._queued -= 1
        self._active += 1
        try:
            return await fn()
        finally:
            self._active -= 1
            self._semaphore.release()
