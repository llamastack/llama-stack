# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import contextlib
import signal
from collections.abc import Iterator
from types import FrameType


class TimeoutError(Exception):
    """Raised when a timed operation exceeds its allowed duration."""

    pass


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Context manager that raises TimeoutError after the specified number of seconds.

    Args:
        seconds: maximum allowed execution time
    """

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
