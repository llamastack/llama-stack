# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .inspect_service import InspectService
from .models import HealthInfo, ListRoutesResponse, RouteInfo, VersionInfo

# Backward compatibility - export Inspect as alias for InspectService
Inspect = InspectService

__all__ = ["Inspect", "InspectService", "ListRoutesResponse", "RouteInfo", "HealthInfo", "VersionInfo"]
