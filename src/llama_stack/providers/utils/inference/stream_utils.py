# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from llama_stack.log import get_logger

log = get_logger(name=__name__, category="providers::utils")


def _normalize_tool_call_arguments(chunk) -> None:
    """Normalize ``arguments=None`` to ``""`` on tool-call delta chunks.

    The OpenAI streaming spec always sends ``arguments=""`` on the first
    tool-call delta, but some providers (vLLM, TGI, etc.) send ``None``.
    Third-party stream wrappers such as opentelemetry-instrumentation-openai-v2
    assume the spec format and crash on ``None``.  Normalizing here keeps
    every downstream consumer safe.
    """
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        for tc in getattr(delta, "tool_calls", None) or []:
            func = getattr(tc, "function", None)
            if func is not None and func.arguments is None:
                func.arguments = ""


async def wrap_async_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Wrap an async stream to ensure it returns a proper AsyncIterator.
    """
    try:
        async for item in stream:
            _normalize_tool_call_arguments(item)
            yield item
    except Exception as e:
        log.error(f"Error in wrapped async stream: {e}")
        raise
