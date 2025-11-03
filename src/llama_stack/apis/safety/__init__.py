# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import (
    ModerationObject,
    ModerationObjectResults,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
    SafetyViolation,
    ViolationLevel,
)
from .safety_service import SafetyService, ShieldStore

# Backward compatibility - export Safety as alias for SafetyService
Safety = SafetyService

__all__ = [
    "Safety",
    "SafetyService",
    "ShieldStore",
    "ModerationObject",
    "ModerationObjectResults",
    "RunShieldRequest",
    "RunShieldResponse",
    "RunModerationRequest",
    "SafetyViolation",
    "ViolationLevel",
]
