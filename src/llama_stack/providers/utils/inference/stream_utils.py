# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


async def wrap_async_stream[T](stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Wrap an async stream to ensure it returns a proper AsyncIterator.
    """
    try:
        async for item in stream:
            yield item
    except Exception as e:
        logger.error(f"Error in wrapped async stream: {e}")
        raise
