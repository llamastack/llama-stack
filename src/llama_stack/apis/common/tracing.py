# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def mark_as_traced(cls):
    """
    Mark a protocol for automatic tracing when telemetry is enabled.

    This is a metadata-only decorator with no dependencies on core.
    Actual tracing is applied by core routers at runtime if telemetry is enabled.

    Usage:
        @runtime_checkable
        @mark_as_traced
        class MyProtocol(Protocol):
            ...
    """
    cls.__marked_for_tracing__ = True
    return cls
