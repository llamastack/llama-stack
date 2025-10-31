# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from contextvars import ContextVar

from llama_stack.core.telemetry.tracing import CURRENT_TRACE_CONTEXT

_MISSING = object()


def preserve_contexts_async_generator[T](
    gen: AsyncGenerator[T, None], context_vars: list[ContextVar]
) -> AsyncGenerator[T, None]:
    """
    Wraps an async generator to preserve context variables across iterations.
    This is needed because we start a new asyncio event loop for each streaming request,
    and we need to preserve the context across the event loop boundary.
    """
    # Capture initial context values
    initial_context_values = {context_var.name: context_var.get() for context_var in context_vars}

    async def wrapper() -> AsyncGenerator[T, None]:
        while True:
            previous_values: dict[ContextVar, object] = {}
            tokens: dict[ContextVar, object] = {}

            # Restore ALL context values before any await and capture previous state
            # This is needed to propagate context across async generator boundaries
            for context_var in context_vars:
                try:
                    previous_values[context_var] = context_var.get()
                except LookupError:
                    previous_values[context_var] = _MISSING
                tokens[context_var] = context_var.set(initial_context_values[context_var.name])

            def _restore_context_var(context_var: ContextVar, *, _tokens=tokens, _prev=previous_values) -> None:
                token = _tokens.get(context_var)
                previous_value = _prev.get(context_var, _MISSING)
                if token is not None:
                    try:
                        context_var.reset(token)
                        return
                    except (RuntimeError, ValueError):
                        pass

                if previous_value is _MISSING:
                    context_var.set(None)
                else:
                    context_var.set(previous_value)

            try:
                item = await gen.__anext__()
            except StopAsyncIteration:
                # Restore all context vars before exiting to prevent leaks
                # Use _restore_context_var for all vars to properly restore to previous values
                for context_var in context_vars:
                    _restore_context_var(context_var)
                break
            except Exception:
                # Restore all context vars on exception
                for context_var in context_vars:
                    _restore_context_var(context_var)
                raise

            try:
                yield item
                # Update our tracked values with any changes made during this iteration
                # Only for non-trace context vars - trace context must persist across yields
                # to allow nested span tracking for telemetry
                for context_var in context_vars:
                    if context_var is not CURRENT_TRACE_CONTEXT:
                        initial_context_values[context_var.name] = context_var.get()
            finally:
                # Restore non-trace context vars after each yield to prevent leaks between requests
                # CURRENT_TRACE_CONTEXT is NOT restored here to preserve telemetry span stack
                for context_var in context_vars:
                    if context_var is not CURRENT_TRACE_CONTEXT:
                        _restore_context_var(context_var)

    return wrapper()
