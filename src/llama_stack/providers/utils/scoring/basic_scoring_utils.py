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
    pass


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
