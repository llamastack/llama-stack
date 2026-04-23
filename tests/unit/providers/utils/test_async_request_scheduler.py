# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import pytest

from ogx.providers.utils.async_request_scheduler import (
    AsyncRequestScheduler,
    QueueFullError,
)


async def test_basic_submit():
    scheduler = AsyncRequestScheduler(max_concurrency=2)
    result = await scheduler.submit(lambda: _immediate("hello"))
    assert result == "hello"
    assert scheduler.active == 0
    assert scheduler.queued == 0


async def test_concurrency_limit_enforced():
    """Verify that at most max_concurrency tasks run simultaneously."""
    scheduler = AsyncRequestScheduler(max_concurrency=2)
    peak_concurrency = 0
    lock = asyncio.Lock()
    barrier = asyncio.Event()

    async def tracked_work():
        nonlocal peak_concurrency
        async with lock:
            peak_concurrency = max(peak_concurrency, scheduler.active)
        await barrier.wait()
        return True

    tasks = [asyncio.create_task(scheduler.submit(lambda: tracked_work())) for _ in range(5)]

    await asyncio.sleep(0.05)
    assert scheduler.active == 2
    assert scheduler.queued == 3
    assert peak_concurrency == 2

    barrier.set()
    results = await asyncio.gather(*tasks)
    assert all(results)
    assert scheduler.active == 0
    assert scheduler.queued == 0


async def test_queue_full_rejected():
    scheduler = AsyncRequestScheduler(max_concurrency=1, max_queue_size=1)
    hold = asyncio.Event()

    async def blocked():
        await hold.wait()

    task1 = asyncio.create_task(scheduler.submit(lambda: blocked()))
    await asyncio.sleep(0.01)
    assert scheduler.active == 1

    task2 = asyncio.create_task(scheduler.submit(lambda: blocked()))
    await asyncio.sleep(0.01)
    assert scheduler.queued == 1

    with pytest.raises(QueueFullError, match="queue is full"):
        await scheduler.submit(lambda: blocked())

    hold.set()
    await task1
    await task2


async def test_fifo_ordering():
    """Queued tasks execute in the order they were submitted."""
    scheduler = AsyncRequestScheduler(max_concurrency=1)
    order: list[int] = []
    gate = asyncio.Event()

    async def record(n: int):
        if n == 0:
            await gate.wait()
        order.append(n)

    tasks = [asyncio.create_task(scheduler.submit(lambda n=i: record(n))) for i in range(4)]
    await asyncio.sleep(0.02)

    gate.set()
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3]


async def test_exception_releases_slot():
    scheduler = AsyncRequestScheduler(max_concurrency=1)

    async def failing():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await scheduler.submit(lambda: failing())

    assert scheduler.active == 0
    result = await scheduler.submit(lambda: _immediate(42))
    assert result == 42


async def test_invalid_config():
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        AsyncRequestScheduler(max_concurrency=0)

    with pytest.raises(ValueError, match="max_queue_size must be >= 0"):
        AsyncRequestScheduler(max_concurrency=1, max_queue_size=-1)


async def test_unlimited_queue():
    scheduler = AsyncRequestScheduler(max_concurrency=1, max_queue_size=0)
    hold = asyncio.Event()

    async def blocked():
        await hold.wait()
        return True

    tasks = [asyncio.create_task(scheduler.submit(lambda: blocked())) for _ in range(20)]
    await asyncio.sleep(0.02)
    assert scheduler.active == 1
    assert scheduler.queued == 19

    hold.set()
    results = await asyncio.gather(*tasks)
    assert len(results) == 20


async def _immediate(value):
    return value
