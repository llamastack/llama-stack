# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shields API protocol and models.

This module contains the Shields protocol definition.
Pydantic models are defined in llama_stack_api.shields.models.
The FastAPI router is defined in llama_stack_api.shields.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Shields

# Import models for re-export
from .models import (
    CommonShieldFields,
    GetShieldRequest,
    ListShieldsResponse,
    RegisterShieldRequest,
    Shield,
    ShieldInput,
    UnregisterShieldRequest,
)

__all__ = [
    "Shields",
    "Shield",
    "ShieldInput",
    "CommonShieldFields",
    "ListShieldsResponse",
    "GetShieldRequest",
    "RegisterShieldRequest",
    "UnregisterShieldRequest",
    "fastapi_routes",
]
