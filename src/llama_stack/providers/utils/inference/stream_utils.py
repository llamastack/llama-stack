# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from llama_stack.log import get_logger

log = get_logger(name=__name__, category="providers::utils")


def _patch_otel_choice_buffer():
    """Patch OTEL's ChoiceBuffer to handle ``arguments=None`` before it
    reaches ``"".join()`` in cleanup.

    TODO: Remove this once https://github.com/open-telemetry/opentelemetry-python-contrib/issues/4344 is fixed.
    """
    try:
        from opentelemetry.instrumentation.openai_v2.patch import ChoiceBuffer  # type: ignore[import-not-found]
    except ImportError:
        return

    _original = ChoiceBuffer.append_tool_call

    def _safe_append(self, tool_call):
        func = getattr(tool_call, "function", None)
        if func is not None and func.arguments is None:
            func.arguments = ""
        _original(self, tool_call)

    ChoiceBuffer.append_tool_call = _safe_append


async def wrap_async_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Wrap an async stream to ensure it returns a proper AsyncIterator.
    """
    try:
        async for item in stream:
            yield item
    except Exception as e:
        log.error(f"Error in wrapped async stream: {e}")
        raise
