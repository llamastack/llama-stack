# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import CommonShieldFields, ListShieldsResponse, RegisterShieldRequest, Shield, ShieldInput
from .shields_service import ShieldsService

# Backward compatibility - export Shields as alias for ShieldsService
Shields = ShieldsService

__all__ = [
    "Shields",
    "ShieldsService",
    "Shield",
    "ShieldInput",
    "CommonShieldFields",
    "ListShieldsResponse",
    "RegisterShieldRequest",
]
