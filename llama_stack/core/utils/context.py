# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from contextvars import ContextVar


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
            try:
                # Restore context values before any await
                for context_var in context_vars:
                    context_var.set(initial_context_values[context_var.name])

                item = await gen.__anext__()

                # Update our tracked values with any changes made during this iteration
                for context_var in context_vars:
                    initial_context_values[context_var.name] = context_var.get()

                yield item

            except StopAsyncIteration:
                break

    return wrapper()
