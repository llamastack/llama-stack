# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import Coroutine
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context


def create_task_with_detached_otel_context(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create an asyncio task that does not inherit the current OpenTelemetry trace context.

    asyncio.create_task copies all contextvars at creation time, which causes
    fire-and-forget or long-lived background tasks to be attributed to whatever
    request happened to spawn them. This inflates trace durations and bundles
    unrelated DB operations under the wrong trace.

    This helper temporarily clears the OTel context before creating the task,
    then immediately restores it so the calling coroutine is unaffected.
    """
    token = otel_context.attach(otel_context.Context())
    try:
        task = asyncio.create_task(coro)
    finally:
        otel_context.detach(token)
    return task


def capture_otel_context() -> otel_context.Context:
    """Snapshot the current OTel context for later use in a different task."""
    return otel_context.get_current()


@contextmanager
def activate_otel_context(ctx: otel_context.Context):
    """Temporarily activate a previously captured OTel context.

    Use this in worker loops that run with a detached (empty) context to
    attribute work back to the originating request's trace.
    """
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)
