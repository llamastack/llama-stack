# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tools API protocol and models.

This module contains the ToolGroups and ToolRuntime protocol definitions.
Pydantic models are defined in llama_stack_api.tools.models.
The FastAPI router is defined in llama_stack_api.tools.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocols for re-export
from .api import ToolGroups, ToolRuntime

# Import models for re-export
from .models import (
    ListToolDefsResponse,
    ListToolGroupsResponse,
    ListToolsRequest,
    SpecialToolGroup,
    ToolDef,
    ToolGroup,
    ToolGroupInput,
    ToolInvocationResult,
    ToolStore,
)

__all__ = [
    "ListToolDefsResponse",
    "ListToolGroupsResponse",
    "ListToolsRequest",
    "SpecialToolGroup",
    "ToolDef",
    "ToolGroup",
    "ToolGroupInput",
    "ToolGroups",
    "ToolInvocationResult",
    "ToolRuntime",
    "ToolStore",
    "fastapi_routes",
]
